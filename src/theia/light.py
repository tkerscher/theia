from __future__ import annotations

import numpy as np
import scipy.constants as consts
from scipy.integrate import quad
from scipy.stats.sampling import NumericalInversePolynomial

import hephaistos as hp
from hephaistos import Program
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import IOQueue

from ctypes import Structure, c_float, c_int64, c_uint32, sizeof
from hephaistos.glsl import buffer_reference, vec2, vec3, vec4
from numpy.ctypeslib import as_array

from theia.lookup import uploadTables
from theia.random import RNG
from theia.scene import Scene
from theia.util import ShaderLoader, compileShader, createPreamble
import theia.units as u

from collections.abc import Callable
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CherenkovLightSource",
    "CherenkovTrackLightSource",
    "ConeLightSource",
    "ConstWavelengthSource",
    "FunctionWavelengthSource",
    "HostLightSource",
    "HostWavelengthSource",
    "LightSampleItem",
    "LightSampler",
    "LightSource",
    "MuonTrackLightSource",
    "ParticleCascadeLightSource",
    "ParticleTrack",
    "PencilLightSource",
    "PolarizedLightSampleItem",
    "SphericalLightSource",
    "UniformWavelengthSource",
    "WavelengthSampleItem",
    "WavelengthSource",
    "frankTamm",
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
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
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
        updateFn: Callable[[HostWavelengthSource, int], None] | None = None,
    ) -> None:
        super().__init__()

        # save params
        self._capacity = capacity
        self._updateFn = updateFn
        # allocate memory
        item = WavelengthSampleItem
        self._queue = IOQueue(item, capacity, mode="update", skipCounter=True)

    @property
    def capacity(self) -> int:
        """Maximum number of samples that can be drawn per run"""
        return self._capacity

    @property
    def queue(self) -> IOQueue:
        """Queue containing the data for the next batch"""
        return self._queue

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(PHOTON_QUEUE_SIZE=self.capacity)
        return preamble + self._sourceCode

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        program.bindParams(PhotonQueueIn=self.queue.tensor)

    # PipelineStage API

    def update(self, i: int) -> None:
        if self._updateFn is not None:
            self._updateFn(self, i)
        super().update(i)

    def run(self, i: int) -> list[hp.Command]:
        return [*self.queue.run(i), *super().run(i)]


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
    normalize: bool, default=True
        If True, adds an extra factor of one over the lambdaRange to the
        contribution to normalize the light source.

    Stage Parameters
    ----------------
    lambdaRange: (float, float), default=(300.0, 700.0)
        min and max wavelength the source emits
    normalize: bool, default=True
        If True, adds an extra factor of one over the lambdaRange to the
        contribution to normalize the light source.
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
        lambdaRange: tuple[float, float] = (300.0, 700.0),
        normalize: bool = True,
    ) -> None:
        super().__init__(
            nRNGSamples=1,
            params={"WavelengthParams": self.SourceParams},
            extra={"normalize"},
        )
        # save params
        self.setParams(lambdaRange=lambdaRange, normalize=normalize)

    # sourceCode via descriptor
    sourceCode = ShaderLoader("wavelengthsource.uniform.glsl")

    @property
    def normalize(self) -> bool:
        """Whether to normalize this source."""
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool) -> None:
        self._normalize = value

    def _finishParams(self, i: int) -> None:
        c = 1.0
        lr = self.getParam("lambdaRange")
        lr = lr[1] - lr[0]
        if lr != 0.0 and not self.normalize:
            c *= abs(lr)
        self.setParam("_contrib", c)


class FunctionWavelengthSource(WavelengthSource):
    """
    Wavelength source generating samples by importance sampling the given
    function or distribution. Numerically integrates and inverts the
    distribution before discretizing the result into a look up table used by
    the GPU.

    Parameters
    ----------
    fn: (float) -> float
        Function to be importance sampled, mapping wavelengths in nm to a
        scalar.
    lambdaRange: (float, float), default=(300.0, 700.0)nm
        Wavelength range of the generated samples.
    numSamples: int, default=1024
        Number of entries in the final look up table.
    """

    name = "Function Wavelength Source"

    class WavelengthParams(Structure):
        _fields_ = [("_table", c_int64), ("_contrib", c_float)]

    def __init__(
        self,
        fn: Callable[[float], float],
        *,
        lambdaRange: tuple[float, float] = (300.0, 700.0) * u.nm,
        numSamples: int = 1024,
    ) -> None:
        super().__init__(
            nRNGSamples=1,
            params={"WavelengthParams": self.WavelengthParams},
        )
        self._updateFn(fn, lambdaRange, numSamples)

    # Source code via descriptor
    sourceCode = ShaderLoader("wavelengthsource.function.glsl")

    def _updateFn(
        self,
        fn: Callable[[float], float],
        lambdaRange: tuple[float, float],
        numSamples: int,
    ) -> None:
        # integrate fn to get constant contribution
        contrib, _ = quad(fn, *lambdaRange)

        class Dist:
            """Bundle fn into a class for use with scipy"""

            def pdf(self, x):
                return fn(x)

        # invert cdf
        inv_cdf = NumericalInversePolynomial(Dist(), domain=lambdaRange)
        # sample inverted cdf
        u = np.linspace(0.0, 1.0, numSamples)
        x = inv_cdf.ppf(u)
        # create table from samples
        self._tableMemory, [table_adr] = uploadTables([x])

        # update params
        self.setParams(_table=table_adr, _contrib=contrib)


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
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
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
    medium: int | None, default=None
        Address of medium the light source is submerged in. If `None` and a
        scene is provided, uses the scene's medium, otherwise a vacuum is
        assumed.
    polarized: bool, default=False
        Whether to save polarization information.
    retrieve: bool, default=True
        Whether the queue gets retrieved from the device after sampling
    scene: Scene | None, default=None
        Optional scene a light may depend on
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
    medium: int
        Address of medium the light source is submerged in. If `None` and a
        scene is provided, uses the scene's medium, otherwise a vacuum is
        assumed.
    """

    name = "Light Sampler"

    class SampleParams(Structure):
        _fields_ = [
            ("count", c_uint32),
            ("baseCount", c_uint32),
            ("medium", buffer_reference),
        ]

    def __init__(
        self,
        source: LightSource,
        wavelengthSource: WavelengthSource,
        capacity: int,
        *,
        rng: RNG | None = None,
        medium: int | None = None,
        polarized: bool = False,
        retrieve: bool = True,
        scene: Scene | None = None,
        batchSize: int = 128,
        code: bytes | None = None,
    ) -> None:
        # Check if we have a RNG if needed
        needRNG = source.nRNGForward > 0 or wavelengthSource.nRNGSamples > 0
        if needRNG and rng is None:
            raise ValueError("Light source requires a RNG but none was given!")
        # init stage
        super().__init__({"SampleParams": self.SampleParams})

        # fetch medium
        if medium is None:
            medium = 0 if scene is None else scene.medium

        # save params
        self._batchSize = batchSize
        self._capacity = capacity
        self._scene = scene
        self._source = source
        self._retrieve = retrieve
        self._polarized = polarized
        self._rng = rng
        self._wavelengthSource = wavelengthSource
        self.setParams(count=capacity, baseCount=0, medium=medium)

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
        self._lightQueue = IOQueue(item, capacity, mode="retrieve")
        item = WavelengthSampleItem
        self._lamQueue = IOQueue(item, capacity, mode="retrieve")
        # bind memory
        self._program.bindParams(
            LightQueueOut=self._lightQueue.tensor,
            PhotonQueueOut=self._lamQueue.tensor,
        )
        if self.scene is not None:
            self.scene.bindParams(self._program)

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
    def lightQueue(self) -> IOQueue:
        """Queue storing the light samples"""
        return self._lightQueue

    @property
    def polarized(self) -> bool:
        """Whether to save polarization information"""
        return self._polarized

    @property
    def retrieve(self) -> bool:
        """Wether the queue gets retrieved from the device after sampling"""
        return self._retrieve

    @property
    def rng(self) -> RNG | None:
        """The random number generator used for sampling. None, if none used."""
        return self._rng

    @property
    def scene(self) -> Scene | None:
        """Optional scene a light may depend on"""
        return self._scene

    @property
    def source(self) -> LightSource:
        """Light source that is sampled"""
        return self._source

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source used to sample wavelengths"""
        return self._wavelengthSource

    @property
    def wavelengthQueue(self) -> IOQueue:
        """Queue holding the sampled wavelengths"""
        return self._lamQueue

    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all stages involved with this sampler in the correct
        order suitable for creating a pipeline.
        """
        stages = [] if self.rng is None else [self.rng]
        stages.extend([self.wavelengthSource, self.source, self])
        return stages

    def run(self, i: int) -> list[hp.Command]:
        self._bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.wavelengthSource.bindParams(self._program, i)

        if self.rng is not None:
            self.rng.bindParams(self._program, i)
        return [
            self._program.dispatch(self._groups),
            *self.lightQueue.run(i),
            *self.wavelengthQueue.run(i),
        ]


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
        updateFn: Callable[[HostLightSource, int], None] | None = None,
    ) -> None:
        super().__init__(supportForward=True, supportBackward=False)

        # save params
        self._capacity = capacity
        self._polarized = polarized
        self._updateFn = updateFn

        # allocate memory
        item = PolarizedLightSampleItem if polarized else LightSampleItem
        self._queue = IOQueue(item, capacity, mode="update", skipCounter=True)

    @property
    def capacity(self) -> int:
        """Maximum number of samples that can be drawn per run"""
        return self._capacity

    @property
    def polarized(self) -> bool:
        """Whether the host also provides polarization information"""
        return self._polarized

    @property
    def queue(self) -> IOQueue:
        """Queue holding the samples for the next batch"""
        return self._queue

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            LIGHT_QUEUE_SIZE=self.capacity,
            LIGHT_QUEUE_POLARIZED=self.polarized,
        )
        return preamble + self._sourceCode

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        program.bindParams(LightQueueIn=self.queue.tensor)

    # PipelineStage API

    def update(self, i: int) -> None:
        if self._updateFn is not None:
            self._updateFn(self, i)
        super().update(i)

    def run(self, i: int) -> list[hp.Command]:
        return [*self.queue.run(i), *super().run(i)]


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
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
        timeRange: tuple[float, float] = (0.0, 100.0) * u.ns,
        cosOpeningAngle: float = 0.5,
        budget: float = 1.0,
        polarizationReference: tuple[float, float, float] = (0.0, 1.0, 0.0),
        stokes: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
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
            ("budget", c_float),
            ("timeRange", vec2),
            ("stokes", vec4),
            ("polarizationRef", vec3),
        ]

    def __init__(
        self,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        timeRange: tuple[float, float] = (0.0, 100.0) * u.ns,
        budget: float = 1.0,
        stokes: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        polarizationRef: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        super().__init__(
            supportForward=True,
            supportBackward=False,
            nRNGForward=1,
            params={"LightParams": self.LightParams},
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
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        timeRange: tuple[float, float] = (0.0, 100.0) * u.ns,
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
        ]

    _sourceCode = ShaderLoader("lightsource.cherenkov.simple.glsl")

    def __init__(
        self,
        *,
        trackStart: tuple[float, float, float] = (0.0, 0.0, 0.0) * u.m,
        trackEnd: tuple[float, float, float] = (100.0, 0.0, 0.0) * u.m,
        startTime: float = 0.0 * u.ns,
        endTime: float = 100.0 * u.m / u.c,
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
        _fields_ = [("track", buffer_reference)]

    _sourceCode = ShaderLoader("lightsource.cherenkov.track.glsl")

    def __init__(
        self,
        track: ParticleTrack | None = None,
        *,
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
        self.setParams(track=track if track is not None else 0)

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


class MuonTrackLightSource(LightSource):
    """
    Light source describing the Cherenkov light produced from a high energy
    muon track and its secondaries particles up to 500 MeV in ice or water as
    described in [1]_.

    Parameters
    ----------
    startPosition: (float, float, float)
        Start position of the track
    startTime: float
        Time at which the muon is at `startPosition`
    endPosition: (float, float, float)
        End position of the track
    endTime: float
        Time at which the muon is at `endPosition`
    muonEnergy: float
        Energy of the muon producing secondary particles
    a_angular: float
        The a parameter of the angular light emission distribution (eq. 4.5 in
        [1]_).
    b_angular: float
        The b parameter of the angular light emission distribution (eq. 4.5 in
        [1]_)
    applyFrankTamm: bool, default=True
        Whether to apply the Frank-Tamm equation describing the light yield as a
        function of wavelength.

    Stage Parameters
    ----------------
    startPosition: (float, float, float)
        Start position of the track
    startTime: float
        Time at which the muon is at `startPosition`
    endPosition: (float, float, float)
        End position of the track
    endTime: float
        Time at which the muon is at `endPosition`
    muonEnergy: float
        Energy of the muon producing secondary particles
    a_angular: float, default=0.39
        The a parameter of the angular light emission distribution (eq. 4.5 in
        [1]_).
    b_angular: float, default=2.61
        The b parameter of the angular light emission distribution (eq. 4.5 in
        [1]_)

    .. [1] L. Raedel "Simulation Studies of the Cherenkov Light Yield from
           Relativistic Particles in High-Energy Neutrino Telescopes with
           Geant 4" (2012)
    """

    name = "Muon Track Light Source"

    class TrackParameters(Structure):
        _fields_ = [
            ("startPosition", vec3),
            ("startTime", c_float),
            ("endPosition", vec3),
            ("endTime", c_float),
            ("_energyScale", c_float),
            ("_a_angular", c_float),
            ("_b_angular", c_float),
        ]

    def __init__(
        self,
        startPosition: tuple[float, float, float],
        startTime: float,
        endPosition: tuple[float, float, float],
        endTime: float,
        muonEnergy: float,
        applyFrankTamm: bool = True,
    ) -> None:
        super().__init__(
            supportForward=True,
            nRNGForward=3,
            supportBackward=True,
            nRNGBackward=1,
            params={"MuonTrackParams": self.TrackParameters},
            extra={"muonEnergy"},
        )

        # set params
        self.setParams(
            startPosition=startPosition,
            startTime=startTime,
            endPosition=endPosition,
            endTime=endTime,
        )
        self.muonEnergy = muonEnergy
        self._applyFrankTamm = applyFrankTamm

    _sourceCode = ShaderLoader("lightsource.particles.muon.glsl")

    @property
    def isFrankTammApplied(self) -> bool:
        """Whether the Frank Tamm equation is applied"""
        return self._applyFrankTamm

    @property
    def muonEnergy(self) -> float:
        """Energy of the muon producing secondary particles"""
        return self._muonEnergy

    @muonEnergy.setter
    def muonEnergy(self, value: float) -> None:
        self._muonEnergy = value
        # calculate energy scale
        self.setParam("_energyScale", 1.1880 + 0.0206 * np.log(value))
        # calculate angular emission profile
        # see notebooks/track_angular_dist_fit.ipynb
        self.setParam("_a_angular", 0.86634 - 7.5624e-3 * np.log10(value))
        self.setParam("_b_angular", 2.5030 + 3.0533e-2 * np.log10(value))

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(FRANK_TAMM_IS=not self.isFrankTammApplied)
        return preamble + self._sourceCode


class ParticleCascadeLightSource(LightSource):
    """
    Light source describing the Cherenkov light emitted by secondary particles
    in electro-magnetic or hadronic showers in water or ice caused by a primary
    particle above 500 MeV as described in [1]_.

    Parameters
    ----------
    startPosition: (float, float, float)
        Start position of the cascade
    startTime: float
        Time point at which the cascade started
    direction: (float, float, float)
        Direction in which the cascade evolves, i.e. away from the start point.
    energyScale: float
        Also called Frank-Tamm factor in [1]_. Ratio of this tracks length to
        one of an Cherenkov track without secondary particles emitting the same
        amount of light. This value is typically larger than one.
    a_angular: float
        The a parameter of the angular light emission distribution (eq. 4.5 in
        [1]_).
    b_angular: float
        The b parameter of the angular light emission distribution (eq. 4.5 in
        [1]_)
    a_long: float
        The a parameter of the longitudinal light emission distribution (eq.
        4.10 in [1]_).
    b_long: float
        The b parameter of the longitudinal light emission distribution. Note
        that this differs from eq. 4.10 in [1]_ in order to match the definition
        used by ice tray. Here (and in ice tray), this is the radiation length
        X_0 divided by the b parameter of the underlying gamma distribution.
    applyFrankTamm: bool, default=True
        Whether to apply the Frank-Tamm equation describing the light yield as a
        function of wavelength.

    Stage Parameters
    ----------------
    startPosition: (float, float, float)
        Start position of the track
    startTime: float
        Time at which the muon is at `startPosition`
    endPosition: (float, float, float)
        End position of the track
    endTime: float
        Time at which the muon is at `endPosition`
    energyScale: float
        Also called Frank-Tamm factor in [1]_. Ratio of this tracks length to
        one of an Cherenkov track without secondary particles emitting the same
        amount of light. This value is typically larger than one.
    a_angular: float
        The a parameter of the angular light emission distribution (eq. 4.5 in
        [1]_).
    b_angular: float
        The b parameter of the angular light emission distribution (eq. 4.5 in
        [1]_)
    a_long: float
        The a parameter of the longitudinal light emission distribution (eq.
        4.10 in [1]_).
    b_long: float
        The b parameter of the longitudinal light emission distribution.

    Note
    ----
    Raedel calculates the radiation length using the following formula:

             1      716.4 [g cm^-2] A
     X_0 = ----- ------------------------
            rho   Z(Z+1) ln(287/sqrt(Z))

    which turns out to be 39.75 cm in ice and 36.08 cm in water [2]_.

    .. [1] L. Raedel "Simulation Studies of the Cherenkov Light Yield from
           Relativistic Particles in High-Energy Neutrino Telescopes with
           Geant 4" (2012)
       [2] L. Raedel, C. Wiebusch: "Calculation of the Cherenkov light yield
           from electromagnetic cascades in ice with Geant4" (2013)
           arXiv:1210.5140v2
    """

    name = "Particle Cascade Light Source"

    class CascadeParameters(Structure):
        _fields_ = [
            ("startPosition", vec3),
            ("startTime", c_float),
            ("direction", vec3),
            ("energyScale", c_float),
            ("a_angular", c_float),
            ("b_angular", c_float),
            ("a_long", c_float),
            ("b_long", c_float),
        ]

    def __init__(
        self,
        startPosition: tuple[float, float, float],
        startTime: float,
        direction: tuple[float, float, float],
        energyScale: float,
        a_angular: float,
        b_angular: float,
        a_long: float,
        b_long: float,
        applyFrankTamm: bool = True,
    ) -> None:
        super().__init__(
            supportForward=True,
            supportBackward=True,
            # unfortunately, the amount of samples drawn is not deterministic
            # because of the rejection algorithm used for the gamma distribution
            # for now we just report a large amount, but this is not nice
            # (with Philox RNG it is not really a problem if we do not shift
            # enough)
            # TODO: fix this
            nRNGForward=12,
            nRNGBackward=10,
            params={"CascadeParams": self.CascadeParameters},
        )

        # save params
        self.setParams(
            startPosition=startPosition,
            startTime=startTime,
            direction=direction,
            energyScale=energyScale,
            a_angular=a_angular,
            b_angular=b_angular,
            a_long=a_long,
            b_long=b_long,
        )
        self._applyFrankTamm = applyFrankTamm

    # source code via descriptor
    _sourceCode = ShaderLoader("lightsource.particles.cascade.glsl")

    @property
    def isFrankTammApplied(self) -> bool:
        """Whether the Frank Tamm equation is applied"""
        return self._applyFrankTamm

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(FRANK_TAMM_IS=not self.isFrankTammApplied)
        return preamble + self._sourceCode


def frankTamm(
    wavelength: ArrayLike,
    refractiveIndex: ArrayLike,
    beta: float = 1.0,
) -> NDArray[np.float64]:
    """Frank Tamm equation
    
    Evaluates the Frank-Tamm equation in units of [m^-1 nm^-1] for the given
    wavelengths and refractive indices:

        d^2           alpha  /            1       \\
     --------- = 2pi ------- | 1 - -------------- |
      dx dlam         lam^2 \\      (beta * n)^2  /

    where alpha is the finestructure constant and beta = v / c the particle's
    speed relative to the speed of light.
    """
    lam = np.asarray(wavelength) / u.nm
    n = beta * np.asarray(refractiveIndex)
    # The 10^9 converts one of the [nm^-1] to [m^-1]
    return 2.0 * np.pi * consts.alpha / lam**2 * (1.0 - (1.0 / n**2)) * 1e9
