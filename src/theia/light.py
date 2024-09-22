from __future__ import annotations

import numpy as np
import hephaistos as hp
from hephaistos import Program
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import QueueBuffer, QueueTensor, QueueView

from ctypes import Structure, c_float, c_uint32, sizeof
from hephaistos.glsl import buffer_reference, vec2, vec3, vec4
from numpy.ctypeslib import as_array

from theia.random import RNG
from theia.util import ShaderLoader, compileShader, createPreamble
import theia.units as u

from typing import Callable, Dict, List, Set, Tuple, Optional
from numpy.typing import NDArray

__all__ = [
    "CherenkovLightSource",
    "CherenkovTrackLightSource",
    "ConeLightSource",
    "ConstWavelengthSource",
    "HostLightSource",
    "HostWavelengthSource",
    "LightSampleItem",
    "LightSampler",
    "LightSource",
    "ParticleTrack",
    "PencilLightSource",
    "PolarizedLightSampleItem",
    "SphericalLightSource",
    "UniformWavelengthSource",
    "WavelengthSampleItem",
    "WavelengthSource",
]


def __dir__():
    return __all__


class WavelengthSource(SourceCodeMixin):
    """
    Base class for samplers producing wavelengths.
    """

    name = "Wavelength Sampler"

    def __init__(
        self,
        *,
        nRNGSamples: int = 0,
        params: Dict[str, type[Structure]] = {},
        extra: Set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self._nRNGSamples = nRNGSamples

    @property
    def nRNGSamples(self) -> int:
        """Amount of random numbers drawn per sample"""
        return self._nRNGSamples


class WavelengthSampleItem(Structure):
    """Structure describing a single wavelength sample."""

    _fields_ = [("wavelength", c_float), ("contrib", c_float)]


class HostWavelengthSource(WavelengthSource):
    """
    Wavelength source passing samples from CPU to GPU.

    Parameters
    ----------
    capacity: int
        Maximum number of samples that can be drawn per run
    updateFn: (HostWavelengthSource, int) -> None | None, default=None
        Optional update function called before the pipeline processes a task.
        `i` is the i-th configuration the update should affect.
        Can be used to stream in new samples on demand.
    """

    _sourceCode = ShaderLoader("wavelengthsource.host.glsl")

    def __init__(
        self,
        capacity: int,
        *,
        updateFn: Optional[Callable[[HostWavelengthSource, int], None]] = None,
    ) -> None:
        super().__init__()

        # save params
        self._capacity = capacity
        self._updateFn = updateFn

        # allocate memory
        self._buffers = [
            QueueBuffer(WavelengthSampleItem, capacity, skipCounter=True)
            for _ in range(2)
        ]
        self._tensor = QueueTensor(WavelengthSampleItem, capacity, skipCounter=True)

    @property
    def capacity(self) -> int:
        """Maximum number of samples that can be drawn per run"""
        return self._capacity

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(PHOTON_QUEUE_SIZE=self.capacity)
        return preamble + self._sourceCode

    def buffer(self, i: int) -> QueueBuffer:
        """Returns the i-th buffer containing the data for the next batch"""
        return self._buffers[i]

    def view(self, i: int) -> QueueView:
        """Returns a view of the data in the i-th buffer"""
        return self.buffer(i).view

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        program.bindParams(PhotonQueueIn=self._tensor)

    # PipelineStage API

    def update(self, i: int) -> None:
        if self._updateFn is not None:
            self._updateFn(self, i)
        super().update(i)

    def run(self, i: int) -> List[hp.Command]:
        return [hp.updateTensor(self.buffer(i), self._tensor), *super().run(i)]


class ConstWavelengthSource(WavelengthSource):
    """
    Sampler generating photons of constant wavelength.

    Parameters
    ----------
    wavelength: float, default=600.0nm
        Constant wavelength

    Stage Parameters
    ----------------
    wavelength: float, default=600.0nm
        Constant wavelength
    """

    name = "Const Wavelength Source"

    class WavelengthParams(Structure):
        _fields_ = [("wavelength", c_float)]

    def __init__(self, wavelength: float = 600.0 * u.nm) -> None:
        super().__init__(params={"WavelengthParams": self.WavelengthParams})
        self.setParams(wavelength=wavelength)

    # source code via descriptor
    sourceCode = ShaderLoader("wavelengthsource.const.glsl")


class UniformWavelengthSource(WavelengthSource):
    """
    Sampler generating photons uniform in wavelength and time.

    Parameters
    ----------
    lambdaRange: (float, float), default=(300.0, 700.0)
        min and max wavelength the source emits

    Stage Parameters
    ----------------
    lambdaRange: (float, float), default=(300.0, 700.0)
        min and max wavelength the source emits
    """

    class SourceParams(Structure):
        _fields_ = [
            ("lambdaRange", vec2),
            ("_contrib", c_float),
        ]

    # lazily load source code
    source_code = None

    def __init__(
        self,
        *,
        lambdaRange: Tuple[float, float] = (300.0, 700.0),
    ) -> None:
        super().__init__(
            nRNGSamples=1,
            params={"WavelengthParams": self.SourceParams},
        )
        # save params
        self.setParams(lambdaRange=lambdaRange)

    # sourceCode via descriptor
    sourceCode = ShaderLoader("wavelengthsource.uniform.glsl")

    def _finishParams(self, i: int) -> None:
        c = 1.0
        lr = self.getParam("lambdaRange")
        lr = lr[1] - lr[0]
        if lr != 0.0:
            c *= abs(lr)
        self.setParam("_contrib", c)


class LightSource(SourceCodeMixin):
    """
    Base class for light sources used in other pipeline stage for generating
    photon packets. Running this stage inside a pipeline updates it usage in
    following stages.
    """

    name = "Light Source"

    def __init__(
        self,
        *,
        supportForward: bool,
        supportBackward: bool,
        nRNGForward: int = 0,
        nRNGBackward: int = 0,
        params: Dict[str, type[Structure]] = {},
        extra: Set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self._forward = supportForward
        self._backward = supportBackward
        self._rngForward = nRNGForward
        self._rngBackward = nRNGBackward

    @property
    def supportForward(self) -> bool:
        """True, if this light source supports forward tracer"""
        return self._forward

    @property
    def supportBackward(self) -> bool:
        """True, if this light source supports backward tracer"""
        return self._backward

    @property
    def nRNGForward(self) -> int:
        """Amount of random numbers drawn per sample in forward mode"""
        return self._rngForward

    @property
    def nRNGBackward(self) -> int:
        """Amount of random numbers drawn per sample in backward mode"""
        return self._rngBackward


class LightSampleItem(Structure):
    """Structure describing a single sample from a light source"""

    _fields_ = [
        ("position", c_float * 3),
        ("direction", c_float * 3),
        ("startTime", c_float),
        ("contrib", c_float),
    ]


class PolarizedLightSampleItem(Structure):
    """
    Structure describing a single polarized sample from a light source.
    Polarization is given by a Stokes' vector and corresponding reference frame
    defined by the direction of vertical polarization of unit length and
    perpendicular to propagation direction.
    """

    _fields_ = [
        ("position", c_float * 3),
        ("direction", c_float * 3),
        ("stokes", c_float * 4),
        ("polarizationRef", c_float * 3),
        ("startTime", c_float),
        ("contrib", c_float),
    ]


class LightSampler(PipelineStage):
    """
    Utility class for sampling a `LightSource` and storing them in a queue.

    Parameters
    ----------
    source: LightSource
        Light source to sample from
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from
    capacity: int
        Maximum number of samples that can be drawn per run
    rng: RNG | None, default=None
        The random number generator used for sampling. May be `None` if `source`
        does not require random numbers.
    polarized: bool, default=False
        Whether to save polarization information.
    retrieve: bool, default=True
        Whether the queue gets retrieved from the device after sampling
    batchSize: int, default=128
        Number of samples drawn per work group
    code: bytes | None, default=None
        Compiled source code. If `None`, the byte code get's compiled from
        source. Note, that the compiled code is not checked. You must ensure
        it matches the configuration.

    Stage Parameters
    ----------------
    count: int, default=capacity
        Number of samples to draw per run. Must be at most `capacity`.
    baseCount: int, default=0
        Offset into the sampling stream of the light source.
    """

    name = "Light Sampler"

    class SampleParams(Structure):
        _fields_ = [
            ("count", c_uint32),
            ("baseCount", c_uint32),
        ]

    def __init__(
        self,
        source: LightSource,
        wavelengthSource: WavelengthSource,
        capacity: int,
        *,
        rng: Optional[RNG] = None,
        polarized: bool = False,
        retrieve: bool = True,
        batchSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        # Check if we have a RNG if needed
        needRNG = source.nRNGForward > 0 or wavelengthSource.nRNGSamples > 0
        if needRNG and rng is None:
            raise ValueError("Light source requires a RNG but none was given!")
        # init stage
        super().__init__({"SampleParams": self.SampleParams})

        # save params
        self._batchSize = batchSize
        self._capacity = capacity
        self._source = source
        self._retrieve = retrieve
        self._polarized = polarized
        self._rng = rng
        self._wavelengthSource = wavelengthSource
        self.setParams(count=capacity, baseCount=0)

        # create code if needed
        if code is None:
            preamble = createPreamble(
                BATCH_SIZE=batchSize,
                LIGHT_QUEUE_SIZE=capacity,
                LIGHT_QUEUE_POLARIZED=polarized,
                PHOTON_QUEUE_SIZE=capacity,
                POLARIZATION=polarized,
            )
            headers = {
                "light.glsl": source.sourceCode,
                "photon.glsl": wavelengthSource.sourceCode,
                "rng.glsl": rng.sourceCode if rng is not None else "",
            }
            code = compileShader("lightsource.sample.glsl", preamble, headers)
        self._code = code
        self._program = hp.Program(self._code)
        # calculate number of workgroups
        self._groups = -(capacity // -batchSize)

        # create queue holding samples
        item = PolarizedLightSampleItem if polarized else LightSampleItem
        self._lightTensor = QueueTensor(item, capacity)
        self._lightBuffer = [
            QueueBuffer(item, capacity) if retrieve else None for _ in range(2)
        ]
        item = WavelengthSampleItem
        self._photonTensor = QueueTensor(item, capacity)
        self._photonBuffer = [
            QueueBuffer(item, capacity) if retrieve else None for _ in range(2)
        ]
        # bind memory
        self._program.bindParams(
            LightQueueOut=self._lightTensor, PhotonQueueOut=self._photonTensor
        )

    @property
    def batchSize(self) -> int:
        """Number of samples drawn per work group"""
        return self._batchSize

    @property
    def capacity(self) -> int:
        """Maximum number of samples that can be drawn per run"""
        return self._capacity

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching"""
        return self._code

    @property
    def lightTensor(self) -> QueueTensor:
        """Tensor holding the queue storing the light samples"""
        return self._lightTensor

    @property
    def polarized(self) -> bool:
        """Whether to save polarization information"""
        return self._polarized

    @property
    def retrieve(self) -> bool:
        """Wether the queue gets retrieved from the device after sampling"""
        return self._retrieve

    @property
    def rng(self) -> Optional[RNG]:
        """The random number generator used for sampling. None, if none used."""
        return self._rng

    @property
    def source(self) -> LightSource:
        """Light source that is sampled"""
        return self._source

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source used to sample wavelengths"""
        return self._wavelengthSource

    @property
    def wavelengthTensor(self) -> QueueTensor:
        """Tensor holding the sampled wavelengths"""
        return self._photonTensor

    def lightBuffer(self, i: int) -> Optional[QueueBuffer]:
        """
        Buffer holding the i-th queue containing light samples.
        `None` if retrieve was set to False.
        """
        return self._lightBuffer[i]

    def lightView(self, i: int) -> Optional[QueueView]:
        """
        View into the i-th queue containing light samples.
        `None` if retrieve was set to `False`.
        """
        return self.lightBuffer(i).view if self.retrieve else None

    def wavelengthBuffer(self, i: int) -> Optional[QueueBuffer]:
        """
        Buffer holding the i-th queue containing wavelength samples.
        `None` if retrieve was set to False.
        """
        return self._photonBuffer[i]

    def wavelengthView(self, i: int) -> Optional[QueueView]:
        """
        View into the i-th queue containing wavelength samples.
        `None` if retrieve was set to False.
        """
        return self.wavelengthBuffer(i).view if self.retrieve else None

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.wavelengthSource.bindParams(self._program, i)

        if self.rng is not None:
            self.rng.bindParams(self._program, i)
        cmds: List[hp.Command] = [self._program.dispatch(self._groups)]
        if self.retrieve:
            cmds.extend(
                [
                    hp.retrieveTensor(self.lightTensor, self.lightBuffer(i)),
                    hp.retrieveTensor(self.wavelengthTensor, self.wavelengthBuffer(i)),
                ]
            )
        return cmds


class HostLightSource(LightSource):
    """
    Light source passing samples from the CPU to the GPU.

    Parameters
    ----------
    capacity: int
        Maximum number of samples that can be drawn per run
    polarized: bool, default=False
        Whether the host also provides polarization information.
    updateFn: (HostLightSource, int) -> None | None, default=None
        Optional update function called before the pipeline processes a task.
        `i` is the i-th configuration the update should affect.
        Can be used to stream in new samples on demand.

    Note
    ----
    To comply with API requirements the shader code expects a wavelength,
    which gets ignored.
    """

    _sourceCode = ShaderLoader("lightsource.host.glsl")

    def __init__(
        self,
        capacity: int,
        *,
        polarized: bool = False,
        updateFn: Optional[Callable[[HostLightSource, int], None]] = None,
    ) -> None:
        super().__init__(supportForward=True, supportBackward=False)

        # save params
        self._capacity = capacity
        self._polarized = polarized
        self._updateFn = updateFn

        # allocate memory
        item = PolarizedLightSampleItem if polarized else LightSampleItem
        self._buffers = [
            QueueBuffer(item, capacity, skipCounter=True) for _ in range(2)
        ]
        self._tensor = QueueTensor(item, capacity, skipCounter=True)

    @property
    def capacity(self) -> int:
        """Maximum number of samples that can be drawn per run"""
        return self._capacity

    @property
    def polarized(self) -> bool:
        """Whether the host also provides polarization information"""
        return self._polarized

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            LIGHT_QUEUE_SIZE=self.capacity,
            LIGHT_QUEUE_POLARIZED=self.polarized,
        )
        return preamble + self._sourceCode

    def buffer(self, i: int) -> QueueBuffer:
        """Returns the i-th buffer containing the data for the next batch"""
        return self._buffers[i]

    def view(self, i: int) -> QueueView:
        """
        Returns a view of the data inside the i-th buffer
        """
        return self.buffer(i).view

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        program.bindParams(LightQueueIn=self._tensor)

    # PipelineStage API

    def update(self, i: int) -> None:
        if self._updateFn is not None:
            self._updateFn(self, i)
        super().update(i)

    def run(self, i: int) -> List[hp.Command]:
        return [hp.updateTensor(self.buffer(i), self._tensor), *super().run(i)]


class ConeLightSource(LightSource):
    """
    Point light source radiating light in a specified cone.
    Can optionally be constant polarized.

    Parameters
    ----------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Position of the light source.
    direction: (float, float, float), default(1.0, 0.0, 0.0)
        Direction of the opening cone.
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    cosOpeningAngle: float, default=0.5
        Cosine of the cones opening angle
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons.
    polarizationReference: (float, float, float), default=(0.0, 1.0, 0.0)
        Polarization reference frame.
    stokes: (float, float, float, float), default=(1.0, 0.0, 0.0, 0.0)
        Specifies the polarization vector.
    polarized: bool, default=True
        Whether the light source should be treated as polarized. If False,
        ignores polarization reference and stoke parameters.

    Stage Parameters
    ----------------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Position of the light source.
    direction: (float, float, float), default(1.0, 0.0, 0.0)
        Direction of the opening cone.
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    cosOpeningAngle: float, default=0.5
        Cosine of the cones opening angle
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons.
    polarizationReference: (float, float, float), default=(0.0, 1.0, 0.0)
        Polarization reference frame.
    stokes: (float, float, float, float), default=(1.0, 0.0, 0.0, 0.0)
        Specifies the polarization vector.

    Note
    ----
    If the light is polarized, the opening angle must not exceed 90 degrees,
    as the polarization reference frame cannot handle this.
    """

    name = "Cone Light Source"

    class LightParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("direction", vec3),
            ("cosOpeningAngle", c_float),
            ("_contribFwd", c_float),
            ("_contribBwd", c_float),
            ("timeRange", vec2),
            ("polarizationReference", vec3),
            ("stokes", vec4),
        ]

    # lazily load source code
    _sourceCode = ShaderLoader("lightsource.cone.glsl")

    def __init__(
        self,
        *,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        timeRange: Tuple[float, float] = (0.0, 100.0) * u.ns,
        cosOpeningAngle: float = 0.5,
        budget: float = 1.0,
        polarizationReference: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        stokes: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        polarized: bool = True,
    ) -> None:
        super().__init__(
            supportForward=True,
            supportBackward=True,
            nRNGForward=3,
            nRNGBackward=1,
            params={"LightParams": ConeLightSource.LightParams},
            extra={"budget"},
        )
        # save params
        self._polarized = polarized
        self.budget = budget
        self.setParams(
            position=position,
            direction=direction,
            timeRange=timeRange,
            cosOpeningAngle=cosOpeningAngle,
            polarizationReference=polarizationReference,
            stokes=stokes,
        )

        # sanity check polarization reference frame
        dir = direction
        polRef = polarizationReference
        if abs(sum(dir[i] * polRef[i] for i in range(3))) > 1.0 - 1e-5:
            raise ValueError("direction and polarizationReference cannot be parallel!")
        if self._polarized and cosOpeningAngle <= 0.0:
            raise ValueError(
                "Opening angles for polarized cone lights must be smaller 90 degrees!"
            )

    @property
    def budget(self) -> float:
        """Total amount of energy or photons the light source distributes among
        the sampled photons."""
        return self._budget

    @budget.setter
    def budget(self, value: float) -> None:
        self._budget = value

    @property
    def polarized(self) -> bool:
        """Whether the light source should be treated as polarized. If False,
        ignores polarization reference and stoke parameters."""
        return self._polarized

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(LIGHTSOURCE_POLARIZED=self.polarized)
        return preamble + self._sourceCode

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        c = self.budget
        tr = self.getParam("timeRange")
        tr = tr[1] - tr[0]
        if tr > 0:
            c *= abs(tr)
        self.setParam("_contribFwd", c)
        # in forward parameter volume cancels with the probability
        # This is not the case in backward mode
        # -> divide contrib by parameter space volume
        c /= 2.0 * np.pi * (1.0 - self.cosOpeningAngle)
        self.setParam("_contribBwd", c)


class PencilLightSource(LightSource):
    """
    Light source generating a pencil beam.

    Parameters
    ----------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Start point of the ray
    direction: (float, float, float), default=(0.0, 0.0, 1.0)
        Direction of the ray
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons.
    stokes: (float, float, float, float), default=(1.0, 0.0, 0.0, 0.0)
        Stokes vector describing polarization. Defaults to unpolarized.
    polarizationRef: (float, float, float), default=(0.0, 1.0, 0.0)
        Reference direction defining the direction of vertically polarized
        light. Must be perpendicular to direction and normalized.

    Stage Parameters
    ----------------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Start point of the ray
    direction: (float, float, float), default=(0.0, 0.0, 1.0)
        Direction of the ray
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons.
    stokes: (float, float, float, float), default=(1.0, 0.0, 0.0, 0.0)
        Stokes vector describing polarization. Defaults to unpolarized.
    polarizationRef: (float, float, float), default=(0.0, 1.0, 0.0)
        Reference direction defining the direction of vertically polarized
        light. Must be perpendicular to direction and normalized.
    """

    name = "Pencil Light Source"

    class LightParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("direction", vec3),
            ("_contrib", c_float),
            ("timeRange", vec2),
            ("stokes", vec4),
            ("polarizationRef", vec3),
        ]

    def __init__(
        self,
        *,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        timeRange: Tuple[float, float] = (0.0, 100.0) * u.ns,
        budget: float = 1.0,
        stokes: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        polarizationRef: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        super().__init__(
            supportForward=True,
            supportBackward=False,
            nRNGForward=1,
            params={"LightParams": self.LightParams},
            extra={"budget"},
        )
        # save params
        self.budget = budget
        self.setParams(
            position=position,
            direction=direction,
            timeRange=timeRange,
            stokes=stokes,
            polarizationRef=polarizationRef,
        )

    @property
    def budget(self) -> float:
        """Total amount of energy or photons the light source distributes among
        the sampled photons"""
        return self._budget

    @budget.setter
    def budget(self, value: float) -> None:
        self._budget = value

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        c = self.budget
        tr = self.getParam("timeRange")
        tr = tr[1] - tr[0]
        if tr > 0:
            c *= abs(tr)
        self.setParam("_contrib", c)

    # lazily load source code
    sourceCode = ShaderLoader("lightsource.pencil.glsl")


class SphericalLightSource(LightSource):
    """
    Isotropic unpolarized point light source, radiating light from a point in
    all direction uniformly.

    Parameters
    ----------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Position the light rays are radiated from.
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons.

    Stage Parameters
    ----------------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Position the light rays are radiated from.
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons.
    """

    name = "Spherical Light Source"

    class LightParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("_contribFwd", c_float),
            ("_contribBwd", c_float),
            ("timeRange", vec2),
        ]

    def __init__(
        self,
        *,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        timeRange: Tuple[float, float] = (0.0, 100.0) * u.ns,
        budget: float = 1.0,
    ) -> None:
        super().__init__(
            supportForward=True,
            supportBackward=True,
            nRNGForward=3,
            nRNGBackward=1,
            params={"LightParams": SphericalLightSource.LightParams},
            extra={"budget"},
        )
        self.budget = budget
        self.setParams(position=position, timeRange=timeRange)

    @property
    def budget(self) -> float:
        """Total amount of energy or photons the light source distributes among
        the sampled photons"""
        return self._budget

    @budget.setter
    def budget(self, value: float) -> None:
        self._budget = value

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        c = self.budget
        tr = self.getParam("timeRange")
        tr = tr[1] - tr[0]
        if tr > 0:
            c *= abs(tr)
        self.setParam("_contribFwd", c)
        # in forward parameter volume cancels with the probability
        # This is not the case in backward mode
        # -> divide contrib by parameter space volume
        c /= 4.0 * np.pi
        self.setParam("_contribBwd", c)

    # lazily load source code via descriptor
    sourceCode = ShaderLoader("lightsource.spherical.glsl")


class CherenkovLightSource(LightSource):
    """
    Light source sampling Cherenkov light from a straight particle track.
    Assumes particle travels at the speed of light (beta = 1.0).

    Parameters
    ----------
    trackStart: (float, float, float), default=(0.0,0.0,0.0)m
        Start position of track
    trackEnd: (float, float, float), default=(100.0,0.0,0.0)m
        End position of track
    startTime: float, default=0.0ns
        Start time of track
    endTime: float, default=100.0 m/c
        End time of track
    medium: int, default=0
        Device address of the medium the particle traverses
    usePhotonCount: bool, default=False
        If `True` sampled radiance has units of number photons, otherwise eV.

    Stage Parameters
    ----------------
    trackStart: (float, float, float), default=(0.0,0.0,0.0)m
        Start position of track
    trackEnd: (float, float, float), default=(100.0,0.0,0.0)m
        End position of track
    startTime: float, default=0.0ns
        Start time of track
    endTime: float, default=100.0 m/c
        End time of track
    medium: int, default=0
        Device address of the medium the particle traverses
    """

    name = "Cherenkov Light Source"

    class LightParams(Structure):
        _fields_ = [
            ("trackStart", vec3),
            ("startTime", c_float),
            ("trackEnd", vec3),
            ("endTime", c_float),
            ("_trackDir", vec3),
            ("_trackDist", c_float),
            ("medium", buffer_reference),
        ]

    _sourceCode = ShaderLoader("lightsource.cherenkov.simple.glsl")

    def __init__(
        self,
        *,
        trackStart: Tuple[float, float, float] = (0.0, 0.0, 0.0) * u.m,
        trackEnd: Tuple[float, float, float] = (100.0, 0.0, 0.0) * u.m,
        startTime: float = 0.0 * u.ns,
        endTime: float = 100.0 * u.m / u.c,
        medium: int = 0,
        usePhotonCount: bool = False,
    ) -> None:
        super().__init__(
            supportForward=True,
            supportBackward=True,
            nRNGForward=2,
            nRNGBackward=0,
            params={"LightParams": self.LightParams},
        )
        # save params
        self._usePhotonCount = usePhotonCount
        self.setParams(
            trackStart=trackStart,
            trackEnd=trackEnd,
            startTime=startTime,
            endTime=endTime,
            medium=medium,
        )

    @property
    def usePhotonCount(self) -> bool:
        """Whether to sample radiance in eV or #photons"""
        return self._usePhotonCount

    @property
    def sourceCode(self) -> str:
        # build preamble
        preamble = createPreamble(FRANK_TAMM_USE_PHOTON_COUNT=self.usePhotonCount)
        # assemble full code
        return preamble + self._sourceCode

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        start = np.array(self.trackStart)
        end = np.array(self.trackEnd)
        dir = end - start
        dist = np.sqrt(np.square(dir).sum(-1))
        dir /= dist
        self.setParams(_trackDir=dir, _trackDist=dist)


class ParticleTrack(hp.ByteTensor):
    """
    Storage class for saving particle tracks on the GPU.

    Parameters
    ----------
    capacity: int
        Maximum number of vertices this track can store
    """

    class Header(Structure):
        """Header describing a particle track"""

        _fields_ = [("length", c_uint32)]

    class Vertex(Structure):
        """Structure describing a single track vertex"""

        _fields_ = [
            ("x", c_float),
            ("y", c_float),
            ("z", c_float),
            ("t", c_float),
        ]

    def __init__(self, capacity: int) -> None:
        size = sizeof(self.Header) + sizeof(self.Vertex) * capacity
        super().__init__(size, mapped=True)
        # check if we managed to map tensor
        if self.memory == 0:
            raise RuntimeError("Could not map tensor in host address space!")

        # save params
        self._capacity = capacity

        # fetch array
        ptr = self.memory
        self._header = self.Header.from_address(ptr)
        ptr += sizeof(self.Header)
        self._arr = (self.Vertex * capacity).from_address(ptr)
        self._flat = (c_float * capacity * 4).from_address(ptr)

        # zero init header
        self.length = 0

    @property
    def capacity(self) -> int:
        """Maximum number of vertices this track can store"""
        return self._capacity

    @property
    def length(self) -> int:
        """Number of segments in the track, i.e. number of vertices - 1"""
        return self._header.length

    @length.setter
    def length(self, value: int) -> None:
        if not 0 <= value <= self.capacity - 1:
            raise ValueError("length must be between 0 and (capacity - 1)")
        self._header.length = value

    def numpy(self, flat: bool = False) -> NDArray:
        """
        Returns a numpy array containing the track vertices.
        If `flat` is `True`, returns an unstructured array of shape (capacity, 4)
        """
        if flat:
            return as_array(self._flat).reshape((-1, 4))
        else:
            return as_array(self._arr)

    def setVertices(self, data: NDArray) -> None:
        """
        Copies the given vertices into the track and updates the corresponding
        properties. Expects data as numpy array of shape (length,4), with
        columns (x[m], y[m], z[m], t[ns]).
        """
        self.numpy(True)[: len(data)] = data
        self.length = len(data) - 1  # #segments


class CherenkovTrackLightSource(LightSource):
    """
    Light source sampling Cherenkov light from a particle track.
    Assumes particle travels at the speed of light (beta = 1.0).

    Parameters
    ----------
    track: ParticleTrack | None
        Track from which Cherenkov light is sampled. Can be set to `None`
        temporarily, but must be set to a valid particle track before sampling
    medium: int, default=0
        Device address of the medium the particle traverses.
    usePhotonCount: bool = False
        Wether to use number of photons as unit of the samples. If `True`
        sampled radiance has energy unit #photons, otherwise `eV`.

    Note
    ----
    Using an invalid particle track or not setting one at all results in
    undefined behavior and may result in the code crashing.
    """

    name = "Cherenkov Track Light Source"

    class TrackParams(Structure):
        _fields_ = [("medium", buffer_reference), ("track", buffer_reference)]

    _sourceCode = ShaderLoader("lightsource.cherenkov.track.glsl")

    def __init__(
        self,
        track: ParticleTrack | None = None,
        *,
        medium: int = 0,
        usePhotonCount: bool = False,
    ) -> None:
        super().__init__(
            supportForward=True,
            supportBackward=False,
            nRNGForward=2,
            params={"TrackParams": self.TrackParams},
        )
        # save params
        self._usePhotonCount = usePhotonCount
        self.setParams(medium=medium, track=track if track is not None else 0)

    @property
    def usePhotonCount(self) -> bool:
        """Whether to sample radiance in eV or #photons"""
        return self._usePhotonCount

    @property
    def sourceCode(self) -> str:
        # build preamble
        preamble = createPreamble(
            FRANK_TAMM_USE_PHOTON_COUNT=self.usePhotonCount,
        )
        # assemble full code
        return preamble + self._sourceCode
