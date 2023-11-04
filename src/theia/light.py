import hephaistos as hp
import warnings

from ctypes import Structure, c_float, c_uint32, addressof
from hephaistos.glsl import vec3, uvec2
from numpy.ctypeslib import as_array
from numpy.typing import NDArray
from typing import Optional, Union, Tuple

from .util import compileShader, loadShader, packUint64


class LightSource:
    """
    Source generating light rays used for tracing.

    Parameters
    ----------
    code: str | bytes
        code used for sampling the light source. Can either be the source code
        or an cached compiled code, which might be differ on different machines
    nPhotons: int, default=4
        number of photons(wavelengths) a single ray carries
    rngSamples: int, default=0
        number of random values drawn from the rng to sample the light
    """

    class Push(Structure):
        """Description of the shader's push constant"""

        _fields_ = [
            ("rayQueue", uvec2),
            ("rngBuffer", uvec2),
            ("medium", uvec2),
            ("count", c_uint32),
            ("rngStride", c_uint32),
        ]

    def __init__(
        self, code: Union[str, bytes], *, nPhotons: int = 4, rngSamples: int = 0
    ) -> None:
        # populate properties
        self._localSize = hp.getSubgroupProperties().subgroupSize
        self._nPhotons = nPhotons
        self._rngSamples = rngSamples

        # compile shader if needed
        if type(code) is str:
            # create shader preamble
            preamble = ""
            preamble += f"#define LOCAL_SIZE {self._localSize}\n"
            preamble += f"#define N_PHOTONS {nPhotons}\n"
            preamble += f"#define RNG_SAMPLES {rngSamples}\n\n"
            # create program
            headers = {"light.glsl": code}
            self._code = compileShader("wavefront.source.glsl", preamble, headers)
        elif type(code) is bytes:
            self._code = code
        else:
            raise RuntimeError("code must be either str or bytes!")
        self._program = hp.Program(self._code)

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching compilation"""
        return self._code

    @property
    def nPhotons(self) -> int:
        """number of photons(wavelengths) a single ray carries"""
        return self._nPhotons

    @property
    def rngSamples(self) -> int:
        """number of random values drawn from the rng to sample the light"""
        return self._rngSamples

    def bindParams(self, **kwargs) -> None:
        """Binds extra params user provided shader code may have defined"""
        self._program.bindParams(**kwargs)

    def sample(
        self,
        count: int,
        queue: Union[hp.Tensor, int],
        rngBuffer: Union[hp.Tensor, int],
        medium: int,
        rngStride: int,
    ) -> hp.Command:
        """
        creates a command for sampling the light source to generate the provided
        amount of samples.

        Parameters
        ----------
        count: int
            amount of samples to generate
        queue: hp.Tensor, int
            either the Tensor or the device address pointing to the memory
            holding the ray queue to populate
        rngBuffer: hp.Tensor, int
            either the Tensor or the device address pointing to the memory
            holding the rng buffer
        medium: int
            device address of the medium the rays start in
        rngStride: int
            stride between each rng stream
        """
        # check if count is multiple of local size
        if count % self._localSize:
            warnings.warn(
                f"count is not a multiple of the device's subgroup size ({self._localSize})"
            )

        # retrieve addresses if needed
        if isinstance(queue, hp.Tensor):
            queue = queue.address
        if isinstance(rngBuffer, hp.Tensor):
            rngBuffer = rngBuffer.address

        # create push constant
        push = self.Push(
            rayQueue=packUint64(queue),
            rngBuffer=packUint64(rngBuffer),
            medium=packUint64(medium),
            count=count,
            rngStride=rngStride,
        )
        # dispatch program
        n = -(count // -self._localSize)  # ceil division
        return self._program.dispatchPush(bytes(push), n)


class HostLightSource(LightSource):
    """
    Light source passing samples from the CPU to the GPU.

    Parameters
    ----------
    maxCount: int
        max number of samples the light source can handle per batch
    nPhotons: int, default=4
        number of photons(wavelengths) a single ray carries
    """

    def __init__(self, maxCount: int, *, nPhotons: int = 4) -> None:
        # create light source
        code = loadShader("lightsource.host.glsl")
        super().__init__(code, nPhotons=nPhotons, rngSamples=0)
        # save param
        self._maxCount = maxCount

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

        # number of floats per row needed for numpy conversion
        self._cols = 6 + nPhotons * 4

        # allocate memory
        self._buffer = hp.ArrayBuffer(SourceRay, maxCount)
        self._tensor = hp.ArrayTensor(SourceRay, maxCount)
        # bind memory
        self.bindParams(Rays=self._tensor)
        # store pointer to underlying float array for numpy conversion
        n = self._cols * maxCount
        self._arr = (c_float * n).from_address(self.address)

    @property
    def address(self) -> int:
        """
        Memory address of the underlying buffer the light rays are stored in.
        Can be used to give external code direct writing access bypassing the
        Python interpreter.
        The memory holds an array of maxCount entries consisting of floats in
        the following layout:

        | position | direction | [ wavelength | radiance | t0 | prob ] * nPhotons
        """
        return addressof(self._buffer.array)

    @property
    def maxCount(self) -> int:
        """max number of samples the light source can handle per batch"""
        return self._maxCount

    def numpy(self, structured: bool = True) -> NDArray:
        """
        Returns the numpy array containing the data for the next batch.
        If structured is True, returns a structured numpy array, else a float
        matrix with one row per ray. Each row has the following layout:

        | position | direction | [ wavelength | radiance | t0 | prob ] * nPhotons
        """
        if structured:
            return self._buffer.numpy()
        else:
            return as_array(self._arr).reshape((-1, self._cols))

    def update(self) -> None:
        """
        Synchronizes the local ray buffer with the GPU. Updating while the light
        source is used by the GPU results in undefined behavior.
        """
        hp.execute(hp.updateTensor(self._buffer, self._tensor))


class SphericalLightSource(LightSource):
    """
    Isotropic light source located at a single point.
    Samples uniform in both time and wavelength.

    Parameters
    ----------
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
    code: bytes | None
        Compiled version of the light source. If None, compiles from source.
        Note, that the compiled code may differ on different machines
    """

    class GLSL(Structure):
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
        *,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        wavelengthRange: Tuple[float, float] = (300.0, 700.0),
        timeRange: Tuple[float, float] = (0.0, 100.0),
        intensity: float = 1.0,
        nPhotons: int = 4,
        code: Optional[bytes] = None,
    ) -> None:
        # load code if needed
        if code is None:
            code = loadShader("lightsource.spherical.glsl")
        # number of samples we'll draw per ray
        rngSamples = 2 + 2 * nPhotons
        super().__init__(code, nPhotons=nPhotons, rngSamples=rngSamples)

        # allocate memory for params
        self._params = hp.StructureBuffer(self.GLSL)
        self._tensor = hp.StructureTensor(self.GLSL)
        # bind tensor
        self.bindParams(LightParams=self._tensor)
        # save params
        self.position = position
        self.wavelengthRange = wavelengthRange
        self.timeRange = timeRange
        self.intensity = intensity
        # upload params
        self.update()

    @property
    def position(self) -> Tuple[float, float, float]:
        """position of the light source"""
        return (
            self._params.position.x,
            self._params.position.y,
            self._params.position.z,
        )

    @position.setter
    def position(self, value: Tuple[float, float, float]) -> None:
        self._params.position.x = value[0]
        self._params.position.y = value[1]
        self._params.position.z = value[2]

    @property
    def wavelengthRange(self) -> Tuple[float, float]:
        """min and max wavelength the source emits"""
        return (
            self._params.lambda_min,
            self._params.lambda_min + self._params.lambdaRange,
        )

    @wavelengthRange.setter
    def wavelengthRange(self, value: Tuple[float, float]) -> None:
        self._params.lambda_min = value[0]
        self._params.lambdaRange = value[1] - value[0]

    @property
    def timeRange(self) -> Tuple[float, float]:
        """start and stop time of the light source"""
        return (self._params.t0, self._params.t0 + self._params.timeDuration)

    @timeRange.setter
    def timeRange(self, value: Tuple[float, float]) -> None:
        self._params.t0 = value[0]
        self._params.timeRange = value[1] - value[0]

    @property
    def intensity(self) -> float:
        """intensity of the light source"""
        return self._intensity

    @intensity.setter
    def intensity(self, value: float) -> None:
        self._intensity = value

    def update(self) -> None:
        """
        Synchronizes the local ray buffer with the GPU. Updating while the light
        source is used by the GPU results in undefined behavior.
        """
        # calculate const contribution
        c = self.intensity
        c /= self._params.lambdaRange
        c /= self._params.timeRange
        # set contribution
        self._params.contribution = c
        # upload to GPU
        hp.execute(hp.updateTensor(self._params, self._tensor))
