from __future__ import annotations

import hephaistos as hp
from hephaistos import Program
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import QueueBuffer, QueueTensor, QueueView

from ctypes import Structure, c_float, c_uint32
from hephaistos.glsl import vec2, vec3

from theia.random import RNG
from theia.util import compileShader, loadShader

from typing import Callable, Dict, List, Set, Tuple, Type, Optional


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
        nLambda: int,
        nRNGSamples: int = 0,
        params: Dict[str, type[Structure]] = {},
        extra: Set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self._nLambda = nLambda
        self._nRNGSamples = nRNGSamples

    @property
    def nLambda(self) -> int:
        """Number of sampled wavelengths in a single light source ray"""
        return self._nLambda

    @property
    def nRNGSamples(self) -> int:
        """Amount of random numbers drawn per sample"""
        return self._nRNGSamples


def createLightSampleItem(nLambda: int) -> Type[Structure]:
    """
    Creates a `Structure` describing the layout of a single light sample
    produced by a `LightSource`.

    Parameters
    ----------
    nLambda: int
        Number of sampled wavelengths in a single light source ray

    Returns
    -------
    item: Type[Structure]
        Layout of a single light source ray
    """

    class LightSampleItem(Structure):
        _fields_ = [
            ("position", c_float * 3),
            ("direction", c_float * 3),
            ("wavelength", c_float * nLambda),
            ("startTime", c_float * nLambda),
            ("contrib", c_float * nLambda),
        ]

    return LightSampleItem


class LightSampler(PipelineStage):
    """
    Utility class for sampling a `LightSource` and storing them in a queue.

    Parameters
    ----------
    source: LightSource
        Light source to sample from
    capacity: int
        Maximum number of samples that can be drawn per run
    rng: RNG | None, default=None
        The random number generator used for sampling. May be `None` if `source`
        does not require random numbers.
    retrieve: bool, default=True
        Wether the queue gets retrieved from the device after sampling
    batchSize: int, default=128
        Number of samples drawn per work group
    code: bytes | None, default=None
        Compiled source code. If `None`, the byte code get's compiled from
        source. Note, that the compiled code is not checked. You must ensure
        it matches the configuration.

    Stage Parameters
    ----------------
    medium: buffer_reference
        Address of the medium the source is embedded in. Zero represents vacuum
    count: int, default=capacity
        Number of samples to draw per run. Must be at most `capacity`.
    baseCount: int, default=0
        Offset into the sampling stream of the light source.

    Example
    -------
    >>> import hephaistos as hp
    >>> import theia.light as l
    >>> from theia.random import PhiloxRNG
    >>> rays = l.SphericalRaySource()
    >>> photons = l.UniformPhotonSource()
    >>> source = l.ModularLight(rays, photons, 4)
    >>> rng = PhiloxRNG()
    >>> sampler = l.LightSampler(source, 8192, rng=rng)
    >>> hp.runPipeline([rng, rays, photons, source, sampler])
    >>> sampler.view(0)
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
        capacity: int,
        *,
        rng: Optional[RNG] = None,
        retrieve: bool = True,
        batchSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        # Check if we have a RNG if needed
        if source.nRNGSamples > 0 and rng is None:
            raise ValueError("Light source requires a RNG but none was given!")
        # init stage
        super().__init__({"SampleParams": self.SampleParams})

        # save params
        self._batchSize = batchSize
        self._capacity = capacity
        self._source = source
        self._retrieve = retrieve
        self._rng = rng
        self.setParams(count=capacity, baseCount=0)

        # create code if needed
        if code is None:
            preamble = ""
            preamble += f"#define LIGHT_QUEUE_SIZE {capacity}\n"
            preamble += f"#define BATCH_SIZE {batchSize}\n"
            preamble += f"#define N_LAMBDA {source.nLambda}\n\n"
            headers = {
                "light.glsl": source.sourceCode,
                "rng.glsl": rng.sourceCode if rng is not None else "",
            }
            code = compileShader("lightsource.sample.glsl", preamble, headers)
        self._code = code
        self._program = hp.Program(self._code)
        # calculate number of workgroups
        self._groups = -(capacity // -batchSize)

        # create queue holding samples
        item = createLightSampleItem(source.nLambda)
        self._tensor = QueueTensor(item, capacity)
        self._buffer = [
            QueueBuffer(item, capacity) if retrieve else None for _ in range(2)
        ]
        # bind memory
        self._program.bindParams(LightQueueOut=self._tensor)

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
    def tensor(self) -> QueueTensor:
        """Tensor holding the queue storing the samples"""
        return self._tensor

    def buffer(self, i: int) -> Optional[QueueBuffer]:
        """
        Buffer holding the i-th queue. `None` if retrieve was set to False.
        """
        return self._buffer[i]

    def view(self, i: int) -> Optional[QueueView]:
        """View into the i-th queue. `None` if retrieve was set to `False`."""
        return self.buffer(i).view if self.retrieve else None

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        if self.rng is not None:
            self.rng.bindParams(self._program, i)
        cmds: List[hp.Command] = [self._program.dispatch(self._groups)]
        if self.retrieve:
            cmds.append(hp.retrieveTensor(self.tensor, self.buffer(i)))
        return cmds


class HostLightSource(LightSource):
    """
    Light source passing samples from the CPU to the GPU.

    Parameters
    ----------
    capacity: int
        Maximum number of samples that can be drawn per run
    nLambda: int, default=4
        Number of sampled wavelengths in a single light source ray
    updateFn: (HostLightSource, int) -> None | None, default=None
        Optional update function called before the pipeline processes a task.
        `i` is the i-th configuration the update should affect.
        Can be used to stream in new samples on demand.
    """

    # cache for source source, lazily loaded (not need if byte code was cached)
    source_code = None

    def __init__(
        self,
        capacity: int,
        *,
        nLambda: int = 4,
        updateFn: Optional[Callable[[HostLightSource, int], None]] = None,
    ) -> None:
        super().__init__(nLambda=nLambda)

        # save params
        self._capacity = capacity
        self._nLambda = nLambda
        self._updateFn = updateFn

        # allocate memory
        item = createLightSampleItem(nLambda)
        self._buffers = [QueueBuffer(item, capacity, skipHeader=True) for _ in range(2)]
        self._tensor = QueueTensor(item, capacity, skipHeader=True)

    @property
    def capacity(self) -> int:
        """Maximum number of samples that can be drawn per run"""
        return self._capacity

    @property
    def nLambda(self) -> int:
        """Number of sampled wavelengths in a single light source ray"""
        return self._nLambda

    @property
    def sourceCode(self) -> str:
        if HostLightSource.source_code is None:
            HostLightSource.source_code = loadShader("lightsource.host.glsl")
        # add preamble
        preamble = f"#define LIGHT_QUEUE_SIZE {self.capacity}\n\n"
        return preamble + HostLightSource.source_code

    def buffer(self, i: int) -> int:
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


class RaySource(SourceCodeMixin):
    """
    Base class for samplers producing blank rays, i.e. a combination of a
    position and a direction.
    """

    name = "Ray Sampler"

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


class DiskRaySource(RaySource):
    """
    Samples point on a disk and creates rays perpendicular to it.

    Parameters
    ----------
    center: (float, float, float), default=(0.0, 0.0, 0.0)
        Center of the disk
    direction: (float, float, float), default=(0.0, 0.0, 1.0)
        Normal of the disk. Rays will travel in this direction.
    radius: float, default=1.0
        Radius of the disk

    Stage Parameters
    ----------------
    center: (float, float, float), default=(0.0, 0.0, 0.0)
        Center of the disk
    direction: (float, float, float), default=(0.0, 0.0, 1.0)
        Normal of the disk. Rays will travel in this direction.
    radius: float, default=1.0
        Radius of the disk
    """

    name = "Disk Ray Sampler"

    class RayParams(Structure):
        _fields_ = [("center", vec3), ("direction", vec3), ("radius", c_float)]

    source_code = None

    def __init__(
        self,
        *,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        radius: float = 1.0,
    ) -> None:
        super().__init__(nRNGSamples=2, params={"RayParams": DiskRaySource.RayParams})
        self.setParams(
            center=center,
            direction=direction,
            radius=radius,
        )

    @property
    def sourceCode(self) -> str:
        if DiskRaySource.source_code is None:
            DiskRaySource.source_code = loadShader("raysource.disk.glsl")
        return DiskRaySource.source_code


class PencilRaySource(RaySource):
    """
    Sampler outputting a constant ray.

    Parameters
    ----------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Start point of the ray
    direction: (float, float, float), default=(0.0, 0.0, 1.0)
        Direction of the ray

    Stage Parameters
    ----------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Start point of the ray
    direction: (float, float, float), default=(0.0, 0.0, 1.0)
        Direction of the ray
    """

    name = "Pencil Beam"

    class RayParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("direction", vec3),
        ]

    source_code = None

    def __init__(
        self,
        *,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        super().__init__(params={"RayParams": PencilRaySource.RayParams})
        self.setParams(position=position, direction=direction)

    @property
    def sourceCode(self) -> str:
        if PencilRaySource.source_code is None:
            PencilRaySource.source_code = loadShader("raysource.pencil.glsl")
        return PencilRaySource.source_code


class SphericalRaySource(RaySource):
    """
    Sampler creating rays uniformly in all direction from a common center.

    Parameters
    ----------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center the rays are emerging from

    Stage Parameters
    ----------------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center the rays are emerging from
    """

    name = "Spherical Ray Sampler"

    class RayParams(Structure):
        _fields_ = [("position", vec3)]

    source_code = None

    def __init__(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__(
            nRNGSamples=2,
            params={"RayParams": SphericalRaySource.RayParams},
        )
        self.setParams(position=position)

    @property
    def sourceCode(self) -> str:
        if SphericalRaySource.source_code is None:
            SphericalRaySource.source_code = loadShader("raysource.spherical.glsl")
        return SphericalRaySource.source_code


class PhotonSource(SourceCodeMixin):
    """
    Base class for samplers producing a single photon sample consisting of a
    wavelength and time point.
    """

    name = "Photon Sampler"

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


class UniformPhotonSource(PhotonSource):
    """
    Sampler generating photons uniform in wavelength and time.

    Parameters
    ----------
    lambdaRange: (float, float), default=(300.0, 700.0)
        min and max wavelength the source emits
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    intensity: float, default=1.0
        intensity of the light source

    Stage Parameters
    ----------------
    lambdaRange: (float, float), default=(300.0, 700.0)
        min and max wavelength the source emits
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    intensity: float, default=1.0
        intensity of the light source
    """

    class SourceParams(Structure):
        _fields_ = [
            ("lambdaRange", vec2),
            ("timeRange", vec2),
            ("_contrib", c_float),
        ]

    # lazily load source code
    source_code = None

    def __init__(
        self,
        *,
        lambdaRange: Tuple[float, float] = (300.0, 700.0),
        timeRange: Tuple[float, float] = (0.0, 100.0),
        intensity: float = 1.0,
    ) -> None:
        super().__init__(
            nRNGSamples=2,
            params={"SourceParams": self.SourceParams},
            extra={"intensity"},
        )
        # save params
        self.setParams(
            lambdaRange=lambdaRange, timeRange=timeRange, intensity=intensity
        )

    @property
    def intensity(self) -> float:
        """Intensity of the light source"""
        return self._intensity

    @intensity.setter
    def intensity(self, value: float) -> None:
        self._intensity = value

    @property
    def sourceCode(self) -> str:
        if UniformPhotonSource.source_code is None:
            UniformPhotonSource.source_code = loadShader("photonsource.uniform.glsl")
        return UniformPhotonSource.source_code

    def _finishParams(self, i: int) -> None:
        # calculate const contribution
        c = self.intensity
        lr = self.getParam("lambdaRange")
        lr = lr[1] - lr[0]
        if lr != 0.0:
            c /= lr
        tr = self.getParam("timeRange")
        tr = tr[1] - tr[0]
        if tr != 0.0:
            c /= tr
        self.setParam("_contrib", c)


class ModularLightSource(LightSource):
    """
    Light source that combines a `RaySource` and a `PhotonSource`.

    Parameters
    ----------
    raySource: RaySource
        Sampler producing rays.
    photonSource: PhotonSource
        Sampler producing single photons
    nLambda: int
        Number of photons to sample
    """

    # lazily load template code
    template_code = None

    def __init__(
        self,
        raySource: RaySource,
        photonSource: PhotonSource,
        nLambda: int,
    ) -> None:
        nRNG = raySource.nRNGSamples + nLambda * photonSource.nRNGSamples
        super().__init__(nLambda=nLambda, nRNGSamples=nRNG)
        self._raySource = raySource
        self._photonSource = photonSource

    @property
    def raySource(self) -> RaySource:
        """Source producing rays"""
        return self._raySource

    @property
    def photonSource(self) -> PhotonSource:
        """Source producing single photons"""
        return self._photonSource

    @property
    def sourceCode(self) -> str:
        # lazily load template code
        if ModularLightSource.template_code is None:
            ModularLightSource.template_code = loadShader("lightsource.modular.glsl")
        # build preamble
        nRNGSource = self.nLambda * self.photonSource.nRNGSamples
        preamble = f"#define RNG_RAY_SAMPLE_OFFSET {nRNGSource}\n"
        # assemble full code
        return "\n".join(
            [
                preamble,
                self.raySource.sourceCode,
                self.photonSource.sourceCode,
                ModularLightSource.template_code,
            ]
        )

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        self.raySource.bindParams(program, i)
        self.photonSource.bindParams(program, i)
