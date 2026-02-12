from __future__ import annotations

import hephaistos as hp

from hephaistos.pipeline import PipelineStage
from hephaistos.queue import IOQueue

from ctypes import Structure, c_float, c_int32, c_uint32, sizeof
from hephaistos.glsl import buffer_reference, mat3, vec3
from theia.util import createCType

from theia.camera import Camera
from theia.compiler import createPreamble, compileShader
from theia.light import LightSource, WavelengthSource
from theia.material import MaterialStore, VACUUM_IDX
from theia.random import RNG
from theia.ray import RayModel
from theia.scene import RectBBox, Transform
from theia.surface import SurfaceModel
from theia.target import LightSourceTarget, Target
import theia.units as u


__all__ = [
    "BackwardLightSampler",
    "CameraDirectSampler",
    "LightSourceTargetSampler",
    "SurfaceInteractionSampler",
    "TargetSampler",
]


def __dir__():
    return __all__


class BackwardLightSampler(PipelineStage):
    """
    Samples the given light source in backward mode.

    Parameters
    ----------
    batchSize: int
        Number of rays to sample per run
    source: LightSource
        Light source to sampler
    ray: RayModel
        Ray model to use
    rng: RNG
        Random number generator
    materials: MaterialStore | None, default = None
        Optional material store containing required medium properties.
    medium: str | int | None, default = None
        Medium from which to sample the light source. Can be specified by either
        its index or its name. Defaults to vacuum if `None`.
    observer: (float, float, float) | None, default=None
        Optional fixed observer position. If `None`, a random observer position
        for each sample will be drawn.
    box_size: float, default=100.0m
        Side length of the cube centered at the origin used to sample the target
        position delegated to the light source.
    """

    class SamplerParams(Structure):
        _fields_ = [
            ("_queueAdr", buffer_reference),
            ("_count", c_uint32),
            ("_mediumIdx", c_uint32),
            ("observer", vec3),
        ]

    def __init__(
        self,
        batchSize: int,
        source: LightSource,
        ray: RayModel,
        *,
        rng: RNG,
        materials: MaterialStore | None = None,
        medium: str | int | None = None,
        observer: tuple[float, float, float] | None = None,
        box_size: float = 100.0 * u.m,
    ) -> None:
        super().__init__(
            {"SamplerParams": BackwardLightSampler.SamplerParams},
            {"medium"},
        )
        # check params
        if not source.supportBackward:
            raise ValueError("Light source does not support backward mode")
        # process params
        if isinstance(medium, str):
            if materials is None:
                raise ValueError("Medium cannot be locked up if no store is provided")
            medium = materials.media[medium]
        elif medium is None:
            medium = VACUUM_IDX
        if materials is None:
            materials = MaterialStore([], mediaSlots=ray.requiredMediumProperties)
        if observer is None:
            observer = (float("NaN"), float("NaN"), float("NaN"))

        # allocate queue
        resultFields = [
            ("observer", c_float * 3),
            ("normal", c_float * 3),
            ("wavelength", c_float),
        ]
        resultFields.extend(ray.forwardItem._fields_)
        item = createCType("Item", resultFields)
        self._queue = IOQueue(item, batchSize, mode="retrieve", skipCounter=True)
        assert self._queue.tensor is not None
        # save params
        self._ray = ray
        self._rng = rng
        self._source = source
        self._materials = materials
        self.setParams(
            _queueAdr=self._queue.tensor.address,
            _count=batchSize,
            _mediumIdx=medium,
            observer=observer,
        )

        # create program
        lamSource = source.wavelengthSource
        preamble = createPreamble(DIM=0.5 * box_size)
        headers = {
            "ray.glsl": ray.sourceCode,
            "rng.glsl": rng.sourceCode,
            "source.glsl": source.sourceCode,
            "photon.glsl": "" if lamSource is None else lamSource.sourceCode,
            **materials.header,
        }
        code = compileShader("lightsource/sample/backward.glsl", preamble, headers)
        self._program = hp.Program(code)
        # bind memory
        materials.bindParams(self._program)

    @property
    def batchSize(self) -> int:
        """Number of samples to draw per batch"""
        return self.getParam("_count")

    @property
    def materials(self) -> MaterialStore:
        """Optional material store in use"""
        return self._materials

    @property
    def medium(self) -> int:
        """Index of the medium the light source is submerged in"""
        return self.getParam("_medium")

    @medium.setter
    def medium(self, value: str | int | None) -> None:
        if self._materials is None and value is not None:
            raise ValueError("Cannot set material if none were provide!")
        if value is None:
            value = VACUUM_IDX
        elif isinstance(value, str):
            assert self._materials is not None  # to make the linter happy...
            value = self._materials.media[value]
        self.setParam("_mediumIdx", value)

    @property
    def queue(self) -> IOQueue:
        """Queue holding the sampled light rays"""
        return self._queue

    @property
    def rayModel(self) -> RayModel:
        """Ray model used for sampling"""
        return self._ray

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    @property
    def source(self) -> LightSource:
        """Source generating light rays"""
        return self._source

    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all stages involved with this sampler in the correct
        order suitable for creating a pipeline.
        """
        stages: list[PipelineStage] = [self.rng]
        if self.source.wavelengthSource is not None:
            stages.append(self.source.wavelengthSource)
        stages.extend([self.source, self])
        return stages

    def run(self, i: int) -> list[hp.Command]:
        for stage in self.collectStages():
            stage.bindParams(self._program, i)
        groups = -(self.batchSize // -512)
        return [
            self._program.dispatch(groups),
            *self.queue.run(i),
        ]


class CameraDirectSampler(PipelineStage):
    """
    Sampler for cameras in direct mode.

    Parameters
    ----------
    batchSize: int
        Number of samples to draw per batch.
    ray: RayModel
        Description of the structure of rays.
    camera: Camera
        Camera to sample
    wavelengthSource: WavelengthSource
        Source from which wavelengths are sampled
    rng: RNG
        Random number generator
    materials: MaterialStore | None, default=None
        Optional material store containing required medium properties. If
        `None`, an empty one with the required slots will be created.
    """

    class SamplerParams(Structure):
        _fields_ = [
            ("_queueAdr", buffer_reference),
            ("_queueSize", c_uint32),
        ]

    def __init__(
        self,
        batchSize: int,
        ray: RayModel,
        camera: Camera,
        wavelengthSource: WavelengthSource,
        *,
        rng: RNG,
        materials: MaterialStore | None = None,
    ) -> None:
        super().__init__({"SamplerParams": CameraDirectSampler.SamplerParams})
        if ray.cameraItem is None:
            raise ValueError("Ray does not define a camera item!")
        if materials is None:
            materials = MaterialStore([], mediaSlots=ray.requiredMediumProperties)

        # allocate queue
        resultFields = [
            ("lightDir", c_float * 3),
            ("samplePos", c_float * 3),
            ("sampleNrm", c_float * 3),
            ("sampleContrib", c_float),
            ("sampleMediumIdx", c_uint32),
        ]
        resultFields.extend(ray.cameraItem._fields_)
        item = createCType("Item", resultFields)
        self._queue = IOQueue(item, batchSize, mode="retrieve", skipCounter=True)
        assert self._queue.tensor is not None
        # save params
        self._ray = ray
        self._rng = rng
        self._camera = camera
        self._photons = wavelengthSource
        self._materials = materials
        self.setParams(_queueAdr=self._queue.tensor.address, _queueSize=batchSize)

        # create program
        headers = {
            "camera.glsl": camera.sourceCode,
            "ray.glsl": ray.sourceCode,
            "rng.glsl": rng.sourceCode,
            "photons.glsl": wavelengthSource.sourceCode,
            **materials.header,
        }
        code = compileShader("camera/sample.direct.glsl", "", headers)
        self._program = hp.Program(code)
        materials.bindParams(self._program)

    @property
    def batchSize(self) -> int:
        """Number of samples to draw per batch"""
        return self.getParam("_queueSize")

    @property
    def camera(self) -> Camera:
        """Camera to sampler"""
        return self._camera

    @property
    def materials(self) -> MaterialStore:
        """Optional material store in use"""
        return self._materials

    @property
    def queue(self) -> IOQueue:
        """Queue holding the sampled light rays"""
        return self._queue

    @property
    def rayModel(self) -> RayModel:
        """Ray model used for sampling"""
        return self._ray

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source for wavelength sampling"""
        return self._photons

    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all stages involved with this sampler in the correct
        order suitable for creating a pipeline.
        """
        return [self.rng, self.wavelengthSource, self.camera, self]

    def run(self, i: int) -> list[hp.Command]:
        for stage in self.collectStages():
            stage.bindParams(self._program, i)
        groups = -(self.batchSize // -512)
        return [
            self._program.dispatch(groups),
            *self.queue.run(i),
        ]


class LightSourceTargetSampler(PipelineStage):
    """
    Sampler for light source targets.

    Parameters
    ----------
    batchSize: int
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
            ("position", c_float * 3),
            ("normal", c_float * 3),
            ("mediumIdx", c_uint32),
            ("contrib", c_float),
        ]

    class SamplerParams(Structure):
        _fields_ = [
            ("_queueAdr", buffer_reference),
            ("_queueSize", c_uint32),
        ]

    def __init__(
        self,
        batchSize: int,
        target: LightSourceTarget,
        wavelengthSource: WavelengthSource,
        *,
        rng: RNG,
    ) -> None:
        super().__init__({"SamplerParams": LightSourceTargetSampler.SamplerParams})

        self._batchSize = batchSize
        self._target = target
        self._photons = wavelengthSource
        self._rng = rng
        self._queue = IOQueue(
            LightSourceTargetSampler.Item, batchSize, mode="retrieve", skipCounter=True
        )
        assert self._queue.tensor is not None
        self.setParams(
            _queueAdr=self._queue.tensor.address,
            _queueSize=batchSize,
        )

        headers = {
            "target.glsl": target.sourceCode,
            "photon.glsl": wavelengthSource.sourceCode,
            "rng.glsl": rng.sourceCode,
        }
        code = compileShader("lightsource/target/sample.glsl", "", headers)
        self._program = hp.Program(code)

    @property
    def batchSize(self) -> int:
        """Number of samples to draw per batch"""
        return self._batchSize

    @property
    def queue(self) -> IOQueue:
        """Queue containing the samples"""
        return self._queue

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

    def run(self, i: int) -> list[hp.Command]:
        for stage in self.collectStages():
            stage.bindParams(self._program, i)
        groups = -(self.batchSize // -512)
        return [
            self._program.dispatch(groups),
            *self.queue.run(i),
        ]


class TargetSampler(PipelineStage):
    """
    Sampler for targets.

    Parameters
    ----------
    batchSize: int
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
            ("observer", c_float * 3),
            ("direction", c_float * 3),
            ("samplePos", c_float * 3),
            ("sampleNrm", c_float * 3),
            ("sampleProb", c_float),
            ("sampleValid", c_uint32),
            ("sampleError", c_int32),
            ("hitPos", c_float * 3),
            ("hitNrm", c_float * 3),
            ("hitProb", c_float),
            ("hitValid", c_uint32),
            ("hitError", c_int32),
            ("occluded", c_uint32),
        ]

    class SamplerParams(Structure):
        _fields_ = [
            ("_queueAdr", buffer_reference),
            ("_queueSize", c_uint32),
            ("_dimMin", vec3),
            ("_dimMax", vec3),
        ]

    def __init__(
        self,
        batchSize: int,
        target: Target,
        *,
        rng: RNG,
        sampleBox: RectBBox | None = None,
    ) -> None:
        super().__init__({"SamplerParams": TargetSampler.SamplerParams})
        if sampleBox is None:
            sampleBox = RectBBox((-20.0, -20.0, -20.0) * u.m, (20.0, 20.0, 20.0) * u.m)

        self._target = target
        self._rng = rng
        self._batch = batchSize
        self._queue = IOQueue(
            TargetSampler.Item, batchSize, mode="retrieve", skipCounter=True
        )
        assert self._queue.tensor is not None
        self.setParams(
            _queueAdr=self._queue.tensor.address,
            _queueSize=batchSize,
            _dimMin=sampleBox.lowerCorner,
            _dimMax=sampleBox.upperCorner,
        )

        headers = {
            "rng.glsl": rng.sourceCode,
            "target.glsl": target.sourceCode,
        }
        code = compileShader("target/sample.glsl", "", headers)
        self._program = hp.Program(code)

    @property
    def batchSize(self) -> int:
        """Number of samples to draw per run"""
        return self._queueSize

    @property
    def queue(self) -> IOQueue:
        """Queue containing samples"""
        return self._queue

    @property
    def rng(self) -> RNG:
        """Random number generator"""
        return self._rng

    @property
    def target(self) -> Target:
        """Target to sample"""
        return self._target

    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all stages involved with this sampler in the correct
        order suitable for creating a pipeline.
        """
        return [self.rng, self.target, self]

    def run(self, i: int) -> list[hp.Command]:
        for stage in self.collectStages():
            stage.bindParams(self._program, i)
        groups = -(self.batchSize // -512)
        return [
            self._program.dispatch(groups),
            *self._queue.run(i),
        ]


class SurfaceInteractionSampler(PipelineStage):
    """
    Samples surface interaction of the given model.

    Parameters
    ----------
    batchSize: int
        Number of samples to draw per run
    ray: RayModel
        Model describing the ray
    surface: SurfaceModel
        Model describing surface interaction
    materials: MaterialStore
        Store containing material data
    rng: RNG
        Random number generator
    source: LightSource | Camera
        Source producing rays.
    wavelengthSource: WavelengthSource | None, default=None
        Sampler for wavelengths to be used with camera
    material: int | str
        Name of index of the surfaces's material
    objectId: int, default=-1
        Object id used to generate hit items
    surfaceNormal: (float, float, float), default(0.0,0.0,1.0)
        Normal of the surface to intersect.
    transform: Transform | None, default=None
        Transformation from object to world space.
    sampleTargetHit: bool, default=False
        Whether to evaluate surface target hits
    """

    name = "Surface Interaction Sampler"

    class SamplerParams(Structure):
        _fields_ = [
            ("_queueAdr", buffer_reference),
            ("_queueSize", c_uint32),
            ("materialIdx", c_uint32),
            ("objectId", c_int32),
            ("surfaceNormal", vec3),
            ("_offset", vec3),
            ("_worldToObj", mat3),
        ]

    def __init__(
        self,
        batchSize: int,
        ray: RayModel,
        surface: SurfaceModel,
        materials: MaterialStore,
        rng: RNG,
        source: LightSource | Camera,
        wavelengthSource: WavelengthSource | None = None,
        *,
        material: int | str,
        objectId: int = -1,
        surfaceNormal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        transform: Transform | None = None,
        sampleTargetHit: bool = False,
    ) -> None:
        super().__init__({"SamplerParams": SurfaceInteractionSampler.SamplerParams})
        self._ray = ray
        self._surface = surface
        self._materials = materials
        self._rng = rng
        self._source = source
        self._photons = wavelengthSource
        if isinstance(material, str):
            material = materials[material]
        if transform is None:
            transform = Transform()
        self.transform = transform

        forward = isinstance(source, LightSource)
        if forward and wavelengthSource is not None:
            raise ValueError("wavelength source can only be specified in backward mode")
        if forward:
            self._photons = source.wavelengthSource
        mode = "forward" if forward else "backward"
        rayItem = ray.forwardItem if forward else ray.backwardItem
        if rayItem is None:
            raise ValueError(f"Ray model does not support {mode} mode")
        srfCode = surface.forwardSourceCode if forward else surface.backwardSourceCode
        if srfCode is None:
            raise ValueError(f"Surface model does not support {mode} mode")

        fields = [
            *((name + "In", t) for name, t in rayItem._fields_),
            *((name + "Out", t) for name, t in rayItem._fields_),
            ("hitResult", c_int32),
        ]
        if sampleTargetHit:
            fields.append(("hitSuccess", c_uint32))
            fields.extend((name + "Hit", t) for name, t in ray.hitItem._fields_)
        item = createCType("Item", fields)
        self._queue = IOQueue(item, batchSize, mode="retrieve", skipCounter=True)
        assert self._queue.tensor is not None

        self.setParams(
            _queueAdr=self._queue.tensor.address,
            _queueSize=batchSize,
            materialIdx=material,
            objectId=objectId,
            surfaceNormal=surfaceNormal,
        )

        preamble = createPreamble(
            SAMPLE_FORWARD=forward,
            RAY_FIELD_COUNT=sizeof(rayItem) // 4,
            SAMPLE_TARGET_HIT=sampleTargetHit,
        )
        headers = {
            "ray.glsl": ray.sourceCode,
            "rng.glsl": rng.sourceCode,
            "source.glsl": source.sourceCode,
            "photon.glsl": "" if self._photons is None else self._photons.sourceCode,
            "surface.glsl": srfCode,
            **materials.header,
        }
        code = compileShader("surface/sample/interaction.glsl", preamble, headers)
        self._program = hp.Program(code)
        materials.bindParams(self._program)

    @property
    def batchSize(self) -> int:
        """Number of samples to draw per run"""
        return self._queueSize

    @property
    def materials(self) -> MaterialStore:
        """Store containing materials to use"""
        return self._materials

    @property
    def queue(self) -> IOQueue:
        """Queue containing samples"""
        return self._queue

    @property
    def rayModel(self) -> RayModel:
        """Model describing rays"""
        return self._ray

    @property
    def rng(self) -> RNG:
        """Random number generator"""
        return self._rng

    @property
    def source(self) -> Camera | LightSource:
        """Source producing rays"""
        return self._source

    @property
    def transform(self) -> Transform:
        """Transformation from object to world space. Must be orthogonal."""
        return self._transform

    @transform.setter
    def transform(self, value: Transform) -> None:
        self._transform = value
        self.setParams(
            _offset=value.offset,
            # GLSL wants transpose (column major), which cancels with inverse
            _worldToObj=value.innerMatrix,
        )

    @property
    def wavelengthSource(self) -> WavelengthSource | None:
        "Optional wavelength source for use with cameras"
        return self._photons

    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all stages involved with this sampler in the correct
        order suitable for creating a pipeline.
        """
        if self._photons is None:
            return [self.rng, self.source, self]
        else:
            return [self.rng, self._photons, self.source, self]

    def run(self, i: int) -> list[hp.Command]:
        for stage in self.collectStages():
            stage.bindParams(self._program, i)
        groups = -(self.batchSize // -512)
        return [
            self._program.dispatch(groups),
            *self._queue.run(i),
        ]
