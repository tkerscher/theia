import hephaistos as hp
import importlib.resources
import warnings

from collections import namedtuple
from ctypes import Structure, c_float, c_uint32, addressof
from hephaistos.glsl import vec3, uvec2
from numpy.ctypeslib import as_array
from numpy.typing import NDArray
from .util import compileShader, loadShader, packUint64


ItemSize = namedtuple("ItemSize", ["const", "perPhoton"])
"""Tuple describing the size of an item inside a single queue"""
def createQueue(item: ItemSize, nPhotons: int, n: int) -> hp.ByteTensor:
    """Creates a tensor with enough space to hold a queue of n items"""
    itemSize = item.const + item.perPhoton * nPhotons
    size = 4 + itemSize * n
    return hp.ByteTensor(size)


class LightSource:
    """
    Source generating light rays used for tracing.

    Parameters
    ----------
    lightCode: str
        shader code used for sampling the light source
    nPhotons: int, default=4
        number of photons(wavelengths) a single ray carries
    rngSamples: int, default=0
        number of random values drawn from the rng to sample the light
    """

    class Push(Structure):
        """Description of the shader's push constant"""
        _fields_ = [
            ("rayQueue", uvec2),
            ("medium", uvec2),
            ("count", c_uint32),
            ("rngStride", c_uint32)
        ]

    def __init__(self, lightCode: str, *, nPhotons: int = 4, rngSamples: int = 0) -> None:
        # populate properties
        self._localSize = hp.getSubgroupProperties().subgroupSize
        self._nPhotons = nPhotons
        self._rngSamples = rngSamples

        # create shader preamble
        preamble = ""
        preamble += f"#define LOCAL_SIZE {self._localSize}\n"
        preamble += f"#define N_PHOTONS {nPhotons}\n"
        preamble += f"#define RNG_SAMPLES {rngSamples}\n\n"
        # create program
        headers = { "light.glsl": lightCode }
        self._code = compileShader("wavefront.source.glsl", preamble, headers)
        self._program = hp.Program(self._code)

    
    @property
    def nPhotons(self) -> int:
        """number of photons(wavelengths) a single ray carries"""
        return self._nPhotons
    
    @property
    def rngSamples(self) -> int:
        """number of random values drawn from the rng to sample the light"""
        return self._rngSamples
    
    def bindParams(self, **kwargs) -> None:
        """Binds extra params to user provided shader code may define"""
        self._program.bindParams(**kwargs)
    
    def sample(self, count: int, queue: hp.ByteTensor, medium: int, rngStride: int) -> hp.Command:
        """
        creates a command for sampling the light source to generate the provided
        amount of samples.

        Parameters
        ----------
        count: int
            amount of samples to generate
        queue: hp.ByteTensor
            tensor holding the ray queue to populate
        medium: int
            device address of the medium the rays start in
        rngStride: int
            stride between each rng stream
        """
        # check if count is multiple of local size
        if count % self._localSize:
            warnings.warn(f"count is not a multiple of the device's subgroup size ({self._localSize})")
        
        # create push constant
        push = self.Push(
            rayQueue=packUint64(queue.address),
            medium=packUint64(medium),
            count=count,
            rngStride=rngStride
        )
        # dispatch program
        n = -(count // -self._localSize) # ceil division
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
                ("log_radiance", c_float),
                ("startTime", c_float),
                ("probability", c_float),
            ]
        class SourceRay(Structure):
            _fields_ = [
                ("position", vec3),
                ("direction", vec3),
                ("photons", SourcePhoton * nPhotons)
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
            return as_array(self._arr).reshape((-1,self._cols))
    
    def update(self) -> None:
        """
        Synchronizes the local ray buffer with the GPU. Updating while the light
        source is used by the GPU results in undefined behavior.
        """
        hp.execute(hp.updateTensor(self._buffer, self._tensor))

