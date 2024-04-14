from __future__ import annotations

import hephaistos as hp
from hephaistos import Program
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import QueueBuffer, QueueTensor, QueueView

from ctypes import Structure, c_float, c_uint32, sizeof
from hephaistos.glsl import buffer_reference, vec2, vec3
from numpy.ctypeslib import as_array

from theia.random import RNG
from theia.util import ShaderLoader, compileShader

from typing import Callable, Dict, List, Set, Tuple, Type, Optional
from numpy.typing import NDArray

__all__ = [
    "CherenkovTrackLightSource",
    "DiskRaySource",
    "HostLightSource",
    "LightSampleItem",
    "LightSampler",
    "LightSource",
    "ModularLightSource",
    "ParticleTrack",
    "PencilRaySource",
    "PhotonSource",
    "RaySource",
    "SphericalRaySource",
    "UniformPhotonSource",
]


def __dir__():
    return __all__


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


class LightSampleItem(Structure):
    _fields_ = [
        ("position", c_float * 3),
        ("direction", c_float * 3),
        ("wavelength", c_float),
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
            preamble += f"#define BATCH_SIZE {batchSize}\n\n"
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
        self._tensor = QueueTensor(LightSampleItem, capacity)
        self._buffer = [
            QueueBuffer(LightSampleItem, capacity) if retrieve else None
            for _ in range(2)
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
    updateFn: (HostLightSource, int) -> None | None, default=None
        Optional update function called before the pipeline processes a task.
        `i` is the i-th configuration the update should affect.
        Can be used to stream in new samples on demand.
    """

    _sourceCode = ShaderLoader("lightsource.host.glsl")

    def __init__(
        self,
        capacity: int,
        *,
        updateFn: Optional[Callable[[HostLightSource, int], None]] = None,
    ) -> None:
        super().__init__()

        # save params
        self._capacity = capacity
        self._updateFn = updateFn

        # allocate memory
        self._buffers = [
            QueueBuffer(LightSampleItem, capacity, skipCounter=True) for _ in range(2)
        ]
        self._tensor = QueueTensor(LightSampleItem, capacity, skipCounter=True)

    @property
    def capacity(self) -> int:
        """Maximum number of samples that can be drawn per run"""
        return self._capacity

    @property
    def sourceCode(self) -> str:
        preamble = f"#define LIGHT_QUEUE_SIZE {self.capacity}\n\n"
        return preamble + self._sourceCode

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

    # source code via descriptor
    sourceCode = ShaderLoader("raysource.disk.glsl")


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

    def __init__(
        self,
        *,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        super().__init__(params={"RayParams": PencilRaySource.RayParams})
        self.setParams(position=position, direction=direction)

    # sourceCode via descriptor
    sourceCode = ShaderLoader("raysource.pencil.glsl")


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

    def __init__(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__(
            nRNGSamples=2,
            params={"RayParams": SphericalRaySource.RayParams},
        )
        self.setParams(position=position)

    # sourceCode via descriptor
    sourceCode = ShaderLoader("raysource.spherical.glsl")


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

    # sourceCode via descriptor
    sourceCode = ShaderLoader("photonsource.uniform.glsl")

    def _finishParams(self, i: int) -> None:
        # calculate const contribution
        # lam ~ U(lam_0, lam_1)
        # t   ~ U(t_0, t_1)
        # => p(t, lam) = 1.0 / (|dLam||dt|) // d(x) = 1.0 if d(x) == 0.0
        # => contrib = L/p = intensity * |dLam|*|dt|
        c = self.intensity
        lr = self.getParam("lambdaRange")
        lr = lr[1] - lr[0]
        if lr != 0.0:
            c *= abs(lr)
        tr = self.getParam("timeRange")
        tr = tr[1] - tr[0]
        if tr != 0.0:
            c *= abs(tr)
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
    """

    # lazily load template code
    _templateCode = ShaderLoader("lightsource.modular.glsl")

    def __init__(
        self,
        raySource: RaySource,
        photonSource: PhotonSource,
    ) -> None:
        nRNG = raySource.nRNGSamples + photonSource.nRNGSamples
        super().__init__(nRNGSamples=nRNG)
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
        # build preamble
        nRNGSource = self.photonSource.nRNGSamples
        preamble = f"#define RNG_RAY_SAMPLE_OFFSET {nRNGSource}\n"
        # assemble full code
        return "\n".join(
            [
                preamble,
                self.raySource.sourceCode,
                self.photonSource.sourceCode,
                self._templateCode,
            ]
        )

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        self.raySource.bindParams(program, i)
        self.photonSource.bindParams(program, i)


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
    photonSource: PhotonSource
        Photon source used to sample wavelengths
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

    class TrackParams(Structure):
        _fields_ = [("medium", buffer_reference), ("track", buffer_reference)]

    _sourceCode = ShaderLoader("lightsource.cherenkov.glsl")

    def __init__(
        self,
        photonSource: PhotonSource,
        track: ParticleTrack | None = None,
        *,
        medium: int = 0,
        usePhotonCount: bool = False,
    ) -> None:
        super().__init__(
            nRNGSamples=2 + photonSource.nRNGSamples,
            params={"TrackParams": self.TrackParams},
        )
        # save params
        self._photonSource = photonSource
        self._usePhotonCount = usePhotonCount
        self.setParams(medium=medium, track=track if track is not None else 0)

    @property
    def photonSource(self) -> PhotonSource:
        """Photon source used to sample wavelengths"""
        return self._photonSource

    @property
    def usePhotonCount(self) -> bool:
        """Wether to sample radiance in eV or #photons"""
        return self._usePhotonCount

    @property
    def sourceCode(self) -> str:
        # build preamble
        preamble = f"#define RNG_RAY_SAMPLE_OFFSET {self.photonSource.nRNGSamples}\n"
        if self.usePhotonCount:
            preamble += f"#define FRANK_TAMM_USE_PHOTON_COUNT 1\n"
        # assemble full code
        return "\n".join([preamble, self.photonSource.sourceCode, self._sourceCode])

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        self.photonSource.bindParams(program, i)
