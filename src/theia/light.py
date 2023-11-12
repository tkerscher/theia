from __future__ import annotations
import hephaistos as hp

from ctypes import Structure, c_float, c_uint32, addressof
from hephaistos.glsl import vec3, uvec2
from numpy.ctypeslib import as_array

from theia.queue import QueueBuffer, QueueTensor, QueueView
from theia.random import RNG
from theia.scheduler import CachedPipelineStage
from .util import compileShader, loadShader, unpackUint64, packUint64

from numpy.typing import NDArray
from typing import Callable, Dict, List, Optional, Union, Tuple, Type


class LightSource(CachedPipelineStage):
    """
    Source generating light rays used for tracing.

    Parameters
    ----------
    capacity: int
        maximum number of samples to draw per run
    code: str | bytes
        code used for sampling the light source. Can either be the source code
        or an cached compiled code, which might be differ on different machines
    rayQueue: hp.Tensor|None, default=None
        tensor containing the ray queue this light source populates.
        must be set before sampling.
    count: int | None, default=None
        number of samples to draw in the next run
        If None, equals to capacity
    nConfigs: int, default=2
        Number of pipeline configurations
    nPhotons: int, default=4
        number of photons(wavelengths) a single ray carries
    medium: int, default=0
        address of the medium the rays starts in. Zero represents vacuum
    rng: RNG, default=None
        generator used for drawing random numbers
    rngSamples: int, default=0
        number of random numbers sampled per ray
    params: { str : Structure }, default={}
        additional parameters
    headers: { str: str }
        extra header files used by the source code
    blockSize: int, default=128
        number of threads in a single local work group (block size in CUDA)
    """

    class SampleParameters(Structure):
        """Description of the shaders params uniform"""

        _fields_ = [
            ("medium", uvec2),
            ("count", c_uint32),
        ]

    def __init__(
        self,
        capacity: int,
        code: Union[str, bytes],
        *,
        count: Optional[int] = None,
        rayQueue: Optional[hp.Tensor] = None,
        nConfigs: int = 2,
        nPhotons: int = 4,
        medium: int = 0,
        rng: Optional[RNG] = None,
        rngSamples: int = 0,
        params: Dict[str, Type[Structure]] = {},
        headers: Dict[str, str] = {},
        blockSize: int = 128,
    ) -> None:
        # collect all params structures
        params = {**params, "SampleParams": self.SampleParameters}
        super().__init__(params, nConfigs)
        # calculate number of work groups per batch
        self._groupSize = -(capacity // -blockSize)  # ceil division

        # compile shader if needed
        if type(code) is str:
            # create shader preamble
            preamble = ""
            if rng is None:
                preamble += "#define NO_RNG 1\n"
            preamble += f"#define LOCAL_SIZE {blockSize}\n"
            preamble += f"#define N_PHOTONS {nPhotons}\n\n"
            # create program
            headers = {"light.glsl": code, **headers}
            if rng is not None:
                headers["rng.glsl"] = rng.sourceCode
            self._code = compileShader("lightsource.sample.glsl", preamble, headers)
        elif type(code) is bytes:
            self._code = code
        else:
            raise RuntimeError("code must be either str or bytes!")
        # create program
        self._program = hp.Program(self._code)

        # save params
        self._capacity = capacity
        self._nPhotons = nPhotons
        self._rng = rng
        self._rngSamples = rngSamples
        self.count = count if count is not None else capacity
        self.medium = medium
        if rayQueue is not None:
            self.rayQueue = rayQueue
        else:
            self._rayQueue = None

    @property
    def capacity(self) -> int:
        """maximum number of light samples processed per run"""
        return self._capacity

    @property
    def count(self) -> int:
        """number of samples to draw in the next run"""
        return self._getParam("SampleParams", "count")

    @count.setter
    def count(self, value: int) -> None:
        self._setParam("SampleParams", "count", value)

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching compilation"""
        return self._code

    @property
    def nPhotons(self) -> int:
        """number of photons(wavelengths) a single ray carries"""
        return self._nPhotons

    @property
    def medium(self) -> int:
        """address of the medium the rays starts in"""
        return unpackUint64(self._getParam("SampleParams", "medium"))

    @medium.setter
    def medium(self, value) -> int:
        self._setParam("SampleParams", "medium", packUint64(value))

    @property
    def rayQueue(self) -> hp.Tensor:
        """
        Tensor containing the ray queue the light source populates.
        Only affects run commands created afterwards.
        """
        return self._rayQueue

    @rayQueue.setter
    def rayQueue(self, value: hp.Tensor) -> None:
        self._rayQueue = value
        self._program.bindParams(RayQueue=value)

    @property
    def rng(self) -> Optional[RNG]:
        """Returns the generator used for drawing random numbers"""
        return self._rng

    @property
    def rngSamples(self) -> int:
        """Amount of random numbers sampled per ray"""
        return self._rngSamples

    def bindParams(self, **kwargs) -> None:
        """Binds extra params user provided shader code may have defined"""
        self._program.bindParams(**kwargs)

    # pipeline stage api

    def run(self, i: int) -> List[hp.Command]:
        self._bindConfigs(self._program, i)
        if self.rng is not None:
            self.rng.bindParams(self._program, i)
        return [self._program.dispatch(self._groupSize)]


class HostLightSource(LightSource):
    """
    Light source passing samples from the CPU to the GPU.

    Parameters
    ----------
    capacity: int
        maximum number of samples to draw per run
    rayQueue: hp.Tensor|None, default=None
        tensor containing the ray queue this light source populates.
        must be set before sampling.
    nBuffers: int, default=2
        number of local buffers
    nPhotons: int, default=4
        number of photons(wavelengths) a single ray carries
    medium: int, default=0
        address of the medium the rays starts in. Zero represents vacuum
    updateFn: (source: HostLightSource, i: int) -> None | None, default=None
        Optional update function called before the pipeline processes a task.
        i is the i-th configuration the update should affect.
    blockSize: int, default=128
        number of threads in a single local work group (block size in CUDA)
    code: bytes | None, default=None
        Compiled version of the light source. If None, compiles from source.
        Note, that the compiled code may differ on different machines
    """

    def __init__(
        self,
        capacity: int,
        *,
        rayQueue: Optional[hp.Tensor] = None,
        nBuffers: int = 2,
        nPhotons: int = 4,
        medium: int = 0,
        updateFn: Optional[Callable[[HostLightSource, int], None]] = None,
        blockSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        # load code if needed
        if code is None:
            code = loadShader("lightsource.host.glsl")
        # create light source
        super().__init__(
            capacity,
            code,
            rayQueue=rayQueue,
            nConfigs=nBuffers,
            nPhotons=nPhotons,
            medium=medium,
            blockSize=blockSize,
        )

        # save updateFn
        self._updateFn = updateFn

        # define GLSL types (depends on nPhotons)
        class SourcePhoton(Structure):
            _fields_ = [
                ("wavelength", c_float),
                ("startTime", c_float),
                ("lin_contrib", c_float),
                ("log_contrib", c_float),
            ]

        class SourceRay(Structure):
            _fields_ = [
                ("position", vec3),
                ("direction", vec3),
                ("photons", SourcePhoton * nPhotons),
            ]

        # allocate memory
        self._buffers = [
            QueueBuffer(SourceRay, capacity, skipHeader=True) for _ in range(nBuffers)
        ]
        self._tensor = QueueTensor(SourceRay, capacity, skipHeader=True)
        # bind memory
        self.bindParams(Rays=self._tensor)

    @property
    def address(self, i: int) -> int:
        """
        Memory address of the i-th underlying buffer the light rays are stored
        in. Can be used to give external code direct writing access bypassing
        the Python interpreter.
        The memory holds an array of maxCount entries consisting of floats in
        the following layout:

        | position | direction | [ wavelength | radiance | t0 | prob ] * nPhotons
        """
        return self._buffers[i].address

    def buffer(self, i: int) -> QueueBuffer:
        """
        Returns the i-th containing the data for the next batch.
        """
        return self._buffers[i]

    def view(self, i: int) -> QueueView:
        """
        Returns a view of the data inside the i-th buffer
        """
        return self.buffer(i).view

    # pipeline stage api

    def update(self, i: int) -> None:
        # first call updateFn as it may alter the source params
        if self._updateFn is not None:
            self._updateFn(self, i)
        super().update(i)

    def run(self, i: int) -> List[hp.Command]:
        return [hp.updateTensor(self._buffers[i], self._tensor), *super().run(i)]


class SphericalLightSource(LightSource):
    """
    Isotropic light source located at a single point.
    Samples uniform in both time and wavelength.

    Parameters
    ----------
    capacity: int
        maximum number of samples to draw per run
    rng: RNG
        random number generator used for sampling
    rayQueue: hp.Tensor|None, default=None
        tensor containing the ray queue this light source populates.
        must be set before sampling.
    position: (float,float,float), default=(0.0, 0.0, 0.0)
        position of the light source
    wavelengthRange: (float, float), default=(300.0, 700.0)
        min and max wavelength the source emits
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    intensity: float, default=1.0
        intensity of the light source
    nPhotons: int, default=4
        number of photons(wavelengths) a single ray carries
    medium: int, default=0
        address of the medium the rays starts in. Zero represents vacuum
    blockSize: int, default=128
        number of threads in a single local work grouop (block size in CUDA)
    code: bytes | None, default=None
        Compiled version of the light source. If None, compiles from source.
        Note, that the compiled code may differ on different machines
    """

    class LightParams(Structure):
        """Params structure"""

        _fields_ = [
            ("position", vec3),
            ("lambda_min", c_float),
            ("lambdaRange", c_float),
            ("t0", c_float),
            ("timeRange", c_float),
            ("contribution", c_float),
        ]

    def __init__(
        self,
        capacity: int,
        *,
        rng: RNG,
        rayQueue: Optional[hp.Tensor] = None,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        wavelengthRange: Tuple[float, float] = (300.0, 700.0),
        timeRange: Tuple[float, float] = (0.0, 100.0),
        intensity: float = 1.0,
        nConfigs: int = 2,
        nPhotons: int = 4,
        medium: int = 0,
        blockSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        # load code if needed
        if code is None:
            code = loadShader("lightsource.spherical.glsl")
        # number of samples we'll draw per ray
        rngSamples = 2 + 2 * nPhotons
        super().__init__(
            capacity,
            code,
            rayQueue=rayQueue,
            nConfigs=nConfigs,
            nPhotons=nPhotons,
            rng=rng,
            rngSamples=rngSamples,
            params={"LightParams": self.LightParams},
            medium=medium,
            blockSize=blockSize,
        )

        # save params
        self.position = position
        self.intensity = intensity  # ! must be first !
        self.wavelengthRange = wavelengthRange
        self.timeRange = timeRange

    # helper for updating contribution
    def _updateContribution(self) -> None:
        # calculate const contribution
        c = self.intensity
        lr = self._getParam("LightParams", "lambdaRange")
        tr = self._getParam("LightParams", "timeRange")
        if lr != 0.0:
            c /= lr
        if tr != 0.0:
            c /= tr
        # set contribution
        self._setParam("LightParams", "contribution", c)

    @property
    def position(self) -> Tuple[float, float, float]:
        """position of the light source"""
        pos = self._getParam("LightParams", "position")
        return (pos.x, pos.y, pos.z)

    @position.setter
    def position(self, value: Tuple[float, float, float]) -> None:
        pos = self._getParam("LightParams", "position")
        pos.x, pos.y, pos.z = value

    @property
    def wavelengthRange(self) -> Tuple[float, float]:
        """min and max wavelength the source emits"""
        lambdaMin = self._getParam("LightParams", "lambda_min")
        lambdaRange = self._getParam("LightParams", "lambdaRange")
        return (lambdaMin, lambdaMin + lambdaRange)

    @wavelengthRange.setter
    def wavelengthRange(self, value: Tuple[float, float]) -> None:
        self._setParam("LightParams", "lambda_min", value[0])
        self._setParam("LightParams", "lambdaRange", value[1] - value[0])
        self._updateContribution()

    @property
    def timeRange(self) -> Tuple[float, float]:
        """start and stop time of the light source"""
        return (self._params.t0, self._params.t0 + self._params.timeDuration)

    @timeRange.setter
    def timeRange(self, value: Tuple[float, float]) -> None:
        self._setParam("LightParams", "t0", value[0])
        self._setParam("LightParams", "timeRange", value[1] - value[0])
        self._updateContribution()

    @property
    def intensity(self) -> float:
        """intensity of the light source"""
        return self._getParam("LightParams", "intensity")

    @intensity.setter
    def intensity(self, value: float) -> None:
        self._setParam("LightParams", "intensity", value)
        self._updateContribution()