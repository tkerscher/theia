import hephaistos as hp
from hephaistos.pipeline import PipelineStage

from theia.camera import Camera
from theia.light import LightSource, WavelengthSource
from theia.material import (
    HenyeyGreensteinPhaseFunction,
    KokhanovskyOceanWaterPhaseMatrix,
    MediumModel,
    WaterBaseModel,
)
from theia.random import RNG
from theia.scene import RectBBox
from theia.target import LightSourceTarget, Target, TargetGuide
from theia.util import createPreamble, compileShader

import theia.units as u

from ctypes import Structure, c_float, c_int32, c_uint32
from hephaistos.glsl import mat4, vec3, vec4
from numpy.typing import NDArray
from numpy.lib.recfunctions import structured_to_unstructured

__all__ = [
    "BackwardLightSampler",
    "CameraDirectSampler",
    "TargetGuideSampler",
    "TargetSampler",
    "WaterTestModel",
]


def __dir__():
    return __all__


def _createResults(data, item):
    """helper function creating a dict of arrays from a buffer"""
    results = {}
    for name, T in item._fields_:
        if issubclass(T, (c_float, c_int32, c_uint32)):
            results[name] = data[name]
        else:
            results[name] = structured_to_unstructured(data[name])
    return results


class BackwardLightSampler(PipelineStage):
    """
    Test program for sampling light in backward mode.

    Parameters
    ----------
    capacity: int
        Number of rays to sample per run
    source: LightSource
        Light source to be sampled
    wavelengthSource: WavelengthSource
        Source producing wavelengths
    rng: RNG
        Random number generator
    box_size: float, default=100.0m
        Side length of the cube centered at the origin used to sample the target
        position delegated to the light source.
    polarized: bool, default=False
        Whether to sample polarized rays
    """

    class PolarizedItem(Structure):
        _fields_ = [
            ("observer", vec3),
            ("normal", vec3),
            ("wavelength", c_float),
            ("position", vec3),
            ("direction", vec3),
            ("stokes", vec4),
            ("polRef", vec3),
            ("startTime", c_float),
            ("contrib", c_float),
        ]

    class UnpolarizedItem(Structure):
        _fields_ = [
            ("observer", vec3),
            ("normal", vec3),
            ("wavelength", c_float),
            ("position", vec3),
            ("direction", vec3),
            ("startTime", c_float),
            ("contrib", c_float),
        ]

    def __init__(
        self,
        capacity: int,
        source: LightSource,
        wavelengthSource: WavelengthSource,
        *,
        rng: RNG,
        box_size: float = 100.0 * u.m,
        polarized: bool = False,
    ) -> None:
        if not source.supportBackward:
            raise ValueError("Light source does not support backward mode!")

        super().__init__()

        self._source = source
        self._rng = rng
        self._photons = wavelengthSource
        self._polarized = polarized
        self._batch = capacity
        self._groups = -(capacity // -32)

        self._item = self.PolarizedItem if polarized else self.UnpolarizedItem
        self._tensor = hp.ArrayTensor(self._item, capacity)
        self._buffer = [hp.ArrayBuffer(self._item, capacity) for _ in range(2)]

        preamble = createPreamble(
            BATCH_SIZE=capacity,
            DIM=0.5 * box_size,
            POLARIZATION=polarized,
        )
        headers = {
            "rng.glsl": rng.sourceCode,
            "light.glsl": source.sourceCode,
            "photon.glsl": wavelengthSource.sourceCode,
        }
        code = compileShader("lightsource.sample.bwd.glsl", preamble, headers)
        self._program = hp.Program(code)
        self._program.bindParams(ResultBuffer=self._tensor)

    @property
    def batchSize(self) -> int:
        """Number of samples to draw per batch"""
        return self._batch

    @property
    def source(self) -> LightSource:
        """Source generating light rays"""
        return self._source

    @property
    def polarized(self) -> bool:
        """Whether polarization is enabled"""
        return self._polarized

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source used to sample wavelengths"""
        return self._photons

    def collectStages(self) -> list[PipelineStage]:
        """Returns a list of all used pipeline stages in correct order"""
        return [self.rng, self.wavelengthSource, self.source, self]

    def get(self, i, name) -> NDArray:
        """Returns the specified field of the i-th buffer"""
        return structured_to_unstructured(self._buffer[i].numpy()[name])

    def getResults(self, i):
        """Returns the result of the i-th buffer"""
        data = self._buffer[i].numpy()
        return _createResults(data, self._item)

    def run(self, i: int):
        self._bindParams(self._program, i)
        self._source.bindParams(self._program, i)
        self._rng.bindParams(self._program, i)
        self._photons.bindParams(self._program, i)
        return [
            self._program.dispatch(self._groups),
            hp.retrieveTensor(self._tensor, self._buffer[i]),
        ]


class CameraDirectSampler(PipelineStage):
    """
    Test program for sampling cameras in direct mode.

    Parameters
    ----------
    capacity: int
        Number of samples to draw
    camera: Camera
        Camera to sample
    wavelengthSource: WavelengthSource
        Source producing wavelengths
    rng: RNG
        Random number generator
    polarized: bool, default=False
        Whether to sample polarized rays
    """

    class PolarizedItem(Structure):
        _fields_ = [
            ("wavelength", c_float),
            ("lightDir", vec3),
            ("samplePos", vec3),
            ("sampleNrm", vec3),
            ("sampleContrib", c_float),
            ("sampleHitPos", vec3),
            ("sampleHitNrm", vec3),
            ("rayPos", vec3),
            ("rayDir", vec3),
            ("rayPolRef", vec3),
            ("mueller", mat4),
            ("rayContrib", c_float),
            ("rayTimeDelta", c_float),
            ("hitPolRef", vec3),
            ("hitPos", vec3),
            ("hitDir", vec3),
            ("hitNrm", vec3),
        ]

    class UnpolarizedItem(Structure):
        _fields_ = [
            ("wavelength", c_float),
            ("lightDir", vec3),
            ("samplePos", vec3),
            ("sampleNrm", vec3),
            ("sampleContrib", c_float),
            ("sampleHitPos", vec3),
            ("sampleHitNrm", vec3),
            ("rayPos", vec3),
            ("rayDir", vec3),
            ("rayContrib", c_float),
            ("rayTimeDelta", c_float),
            ("hitPos", vec3),
            ("hitDir", vec3),
            ("hitNrm", vec3),
        ]

    def __init__(
        self,
        capacity: int,
        camera: Camera,
        wavelengthSource: WavelengthSource,
        *,
        rng: RNG,
        polarized: bool = False,
    ) -> None:
        # check camera
        if not camera.supportDirect:
            raise ValueError("Camera does not support direct mode!")

        super().__init__()

        self._camera = camera
        self._rng = rng
        self._photons = wavelengthSource
        self._polarized = polarized
        self._batch = capacity
        self._groups = -(capacity // -32)

        self._item = self.PolarizedItem if polarized else self.UnpolarizedItem
        self._tensor = hp.ArrayTensor(self._item, capacity)
        self._buffer = [hp.ArrayBuffer(self._item, capacity) for _ in range(2)]

        preamble = createPreamble(
            BATCH_SIZE=capacity,
            POLARIZATION=polarized,
        )
        headers = {
            "camera.glsl": camera.sourceCode,
            "photon.glsl": wavelengthSource.sourceCode,
            "rng.glsl": rng.sourceCode,
        }
        code = compileShader("camera.direct.sample.glsl", preamble, headers)
        self._program = hp.Program(code)
        self._program.bindParams(ResultBuffer=self._tensor)

    @property
    def batchSize(self) -> int:
        """Number of samples to draw per batch"""
        return self._batch

    @property
    def camera(self) -> Camera:
        """Source generating camera rays"""
        return self._camera

    @property
    def polarized(self) -> bool:
        """Whether polarization is enabled"""
        return self._polarized

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source used to sample wavelengths"""
        return self._photons

    def collectStages(self) -> list[PipelineStage]:
        """Returns a list of all used pipeline stages in correct order"""
        return [self.rng, self.wavelengthSource, self.camera, self]

    def get(self, i, name) -> NDArray:
        """Returns the specified field of the i-th buffer"""
        return structured_to_unstructured(self._buffer[i].numpy()[name])

    def getResults(self, i):
        """Returns the result of the i-th buffer"""
        data = self._buffer[i].numpy()
        return _createResults(data, self._item)

    def run(self, i: int):
        self._bindParams(self._program, i)
        self._camera.bindParams(self._program, i)
        self._photons.bindParams(self._program, i)
        self._rng.bindParams(self._program, i)
        return [
            self._program.dispatch(self._groups),
            hp.retrieveTensor(self._tensor, self._buffer[i]),
        ]


class LightSourceTargetSampler(PipelineStage):
    """
    Test program for sampling light source targets.

    Parameters
    ----------
    capacity: int
        Number of samples to draw per run
    target: LightSourceTarget
        Target to sample
    wavelengthSource: WavelengthSource
        Source producing wavelengths
    rng: RNG
        Random number generator
    """

    class Item(Structure):
        _fields_ = [
            ("wavelength", c_float),
            ("position", vec3),
            ("normal", vec3),
            ("contrib", c_float),
        ]

    def __init__(
        self,
        capacity: int,
        target: LightSourceTarget,
        wavelengthSource: WavelengthSource,
        *,
        rng: RNG,
    ) -> None:
        super().__init__()

        self._target = target
        self._photons = wavelengthSource
        self._rng = rng
        self._capacity = capacity
        self._groups = -(capacity // -32)

        self._tensor = hp.ArrayTensor(self.Item, capacity)
        self._buffer = [hp.ArrayBuffer(self.Item, capacity) for _ in range(2)]

        preamble = createPreamble(BATCH_SIZE=capacity)
        headers = {
            "target.glsl": target.sourceCode,
            "photon.glsl": wavelengthSource.sourceCode,
            "rng.glsl": rng.sourceCode,
        }
        code = compileShader("lightsource.target.sample.glsl", preamble, headers)
        self._program = hp.Program(code)
        self._program.bindParams(ResultBuffer=self._tensor)

    @property
    def batchSize(self) -> int:
        """Number of samples to draw per batch"""
        return self._capacity

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    @property
    def target(self) -> LightSourceTarget:
        """Target being sampled"""
        return self._target

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source used to sample wavelengths"""
        return self._photons

    def collectStages(self) -> list[PipelineStage]:
        """Returns a list of all used pipeline stages in correct order"""
        return [self.rng, self.wavelengthSource, self.target, self]

    def get(self, i, name) -> NDArray:
        """Returns the specified field of the i-th buffer"""
        return structured_to_unstructured(self._buffer[i].numpy()[name])

    def getResults(self, i):
        """Returns the result of the i-th buffer"""
        data = self._buffer[i].numpy()
        return _createResults(data, self.Item)

    def run(self, i: int):
        self._bindParams(self._program, i)
        self._target.bindParams(self._program, i)
        self._photons.bindParams(self._program, i)
        self._rng.bindParams(self._program, i)
        return [
            self._program.dispatch(self._groups),
            hp.retrieveTensor(self._tensor, self._buffer[i]),
        ]


class TargetGuideSampler(PipelineStage):
    """
    Test program for sampling target guides.

    Parameters
    ----------
    capacity: int
        Number of samples to draw per run
    guide: TargetGuide
        Target guide to be sampled
    rng: RNG
        Random number generator
    sampleBox: RectBBox
        Rectangular box from which the observer position will be sampled.
    """

    class Item(Structure):
        _fields_ = [
            ("observer", vec3),
            ("sampleDir", vec3),
            ("sampleDist", c_float),
            ("sampleProb", c_float),
            ("evalDir", vec3),
            ("evalDist", c_float),
            ("evalProb", c_float),
        ]

    class Push(Structure):
        _fields_ = [
            ("dimMin", vec3),
            ("dimMax", vec3),
        ]

    def __init__(
        self,
        capacity: int,
        guide: TargetGuide,
        *,
        rng: RNG,
        sampleBox: RectBBox = RectBBox((-20.0,) * 3, (20.0,) * 3),
    ) -> None:
        super().__init__()

        self._guide = guide
        self._rng = rng
        self._batch = capacity
        self._groups = -(capacity // -32)

        self._push = self.Push(
            dimMin=sampleBox.lowerCorner,
            dimMax=sampleBox.upperCorner,
        )

        self._tensor = hp.ArrayTensor(self.Item, capacity)
        self._buffer = [hp.ArrayBuffer(self.Item, capacity) for _ in range(2)]

        preamble = createPreamble(BATCH_SIZE=capacity)
        headers = {
            "rng.glsl": rng.sourceCode,
            "target_guide.glsl": guide.sourceCode,
        }
        code = compileShader("target_guide.sample.glsl", preamble, headers)
        self._program = hp.Program(code)
        self._program.bindParams(ResultBuffer=self._tensor)

    @property
    def batchSize(self) -> int:
        """Number of samples to draw per batch"""
        return self._batch

    @property
    def guide(self) -> TargetGuide:
        """Target guide to be sampled"""
        return self._guide

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    def collectStages(self) -> list[PipelineStage]:
        """Returns a list of all used pipeline stages in correct order"""
        return [self.rng, self.guide, self]

    def getResults(self, i):
        data = self._buffer[i].numpy()
        return _createResults(data, self.Item)

    def run(self, i: int):
        self._bindParams(self._program, i)
        self._guide.bindParams(self._program, i)
        self._rng.bindParams(self._program, i)
        return [
            self._program.dispatchPush(bytes(self._push), self._groups),
            hp.retrieveTensor(self._tensor, self._buffer[i]),
        ]


class TargetSampler(PipelineStage):
    """
    Test program for sampling targets.

    Parameters
    ----------
    capacity: int
        Number of samples to draw per run
    target: Target
        Target to draw samples from
    rng: RNG
        Random number generator
    sampleBox: RectBBox
        Rectangular box from which the observer position will be sampled.
    """

    class Item(Structure):
        _fields_ = [
            ("observer", vec3),
            ("direction", vec3),
            ("samplePos", vec3),
            ("sampleNrm", vec3),
            ("sampleProb", c_float),
            ("sampleValid", c_uint32),
            ("sampleError", c_int32),
            ("hitPos", vec3),
            ("hitNrm", vec3),
            ("hitProb", c_float),
            ("hitValid", c_uint32),
            ("hitError", c_int32),
            ("occluded", c_uint32),
        ]

    class Push(Structure):
        _fields_ = [
            ("dimMin", vec3),
            ("dimMax", vec3),
        ]

    def __init__(
        self,
        capacity: int,
        target: Target,
        *,
        rng: RNG,
        sampleBox: RectBBox = RectBBox((-20.0,) * 3, (20.0,) * 3),
    ) -> None:
        super().__init__()

        self._target = target
        self._rng = rng
        self._batch = capacity
        self._groups = -(capacity // -32)

        self._push = self.Push(
            dimMin=sampleBox.lowerCorner,
            dimMax=sampleBox.upperCorner,
        )

        self._tensor = hp.ArrayTensor(self.Item, capacity)
        self._buffer = [hp.ArrayBuffer(self.Item, capacity) for _ in range(2)]

        preamble = createPreamble(BATCH_SIZE=capacity)
        headers = {
            "rng.glsl": rng.sourceCode,
            "target.glsl": target.sourceCode,
        }
        code = compileShader("target.sample.glsl", preamble, headers)
        self._program = hp.Program(code)
        self._program.bindParams(ResultBuffer=self._tensor)

    @property
    def batchSize(self) -> int:
        """Number of samples to draw per batch"""
        return self._batch

    @property
    def target(self) -> Target:
        """Target to be sampled"""
        return self._target

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    def collectStages(self) -> list[PipelineStage]:
        """Returns a list of all used pipeline stages in correct order"""
        return [self.rng, self.target, self]

    def getResults(self, i):
        data = self._buffer[i].numpy()
        return _createResults(data, self.Item)

    def run(self, i: int):
        self._bindParams(self._program, i)
        self._target.bindParams(self._program, i)
        self._rng.bindParams(self._program, i)
        return [
            self._program.dispatchPush(bytes(self._push), self._groups),
            hp.retrieveTensor(self._tensor, self._buffer[i]),
        ]


class WaterTestModel(
    WaterBaseModel,
    HenyeyGreensteinPhaseFunction,
    KokhanovskyOceanWaterPhaseMatrix,
    MediumModel,
):
    """Simple water model used for testing"""

    def __init__(self) -> None:
        WaterBaseModel.__init__(self, 5.0, 1000.0, 35.0)
        HenyeyGreensteinPhaseFunction.__init__(self, 0.6)
        KokhanovskyOceanWaterPhaseMatrix.__init__(
            self, p90=0.66, theta0=0.25, alpha=4.0, xi=25.6  # voss measurement fit
        )

    ModelName = "water"
