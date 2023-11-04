from __future__ import annotations
import hephaistos as hp

from collections import namedtuple
from ctypes import Structure, c_float, c_uint32
from hephaistos.glsl import vec3, uvec2
from numpy.ctypeslib import as_array
from numpy.typing import NDArray
from typing import Optional, Union, Type
from .util import compileShader, packUint64


ItemSize = namedtuple("ItemSize", ["const", "perPhoton"])
"""Tuple describing the size of an item inside a single queue"""


def createQueue(item: ItemSize, nPhotons: int, n: int) -> hp.ByteTensor:
    """Creates a tensor with enough space to hold a queue of n items"""
    itemSize = item.const + item.perPhoton * nPhotons
    size = 4 + itemSize * n
    return hp.ByteTensor(size)


class HistogramIntegrator:
    """
    Integrator producing a light curve by computing the histogram of samples.

    Parameters
    ----------
    code: str | bytes
        Either the source code for the response function or the compiled shader.
        The compiled shader must match the configuration and might differ on
        different machines.
    capacity: int
        Maximum number of samples the integrator can process in a single run.
    nBins: int
        Number of bins in the histogram
    nPhotons: int
        Number of photons in single ray hit
    batchSize: int
        Number of samples processed in a single batch
    """

    class Push(Structure):
        """Struct defining push constants"""

        _fields_ = [
            ("responseQueue", uvec2),
            ("t0", c_float),
            ("binSize", c_float),
            ("detectorId", c_uint32),
        ]

    def __init__(
        self,
        code: Union[str, bytes],
        capacity: int,
        *,
        nBins: int = 256,
        nPhotons: int = 4,
        batchSize: int = 128,
    ) -> None:
        # save params
        self._capacity = capacity
        self._nBins = nBins
        self._nPhotons = nPhotons
        self._batchSize = batchSize

        # allocate local histograms
        self._nHist = -(capacity // -batchSize)  # ceil division
        self._hist = hp.FloatTensor(self._nHist * nBins)

        # compile source code if needed
        if type(code) is str:
            # create preamble
            preamble = ""
            preamble += f"#define BATCH_SIZE {batchSize}\n"
            preamble += f"#define N_BINS {nBins}\n"
            preamble += f"#define N_PHOTONS {nPhotons}\n\n"
            # compile source code
            headers = {"response.glsl": code}
            self._code = compileShader(
                "wavefront.response.hist.glsl", preamble, headers
            )
        elif type(code) is bytes:
            self._code = code
        else:
            raise RuntimeError("code must be either str or bytes!")
        # create program
        self._program = hp.Program(self._code)
        # bind params
        self._program.bindParams(Histograms=self._hist)

    @property
    def batchSize(self) -> int:
        """Number of samples processed in a single batch"""
        return self._batchSize

    @property
    def capacity(self) -> int:
        """Maximum number of samples the integrator can process in a single run"""
        return self._capacity

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching compilation"""
        return self._code

    @property
    def histograms(self) -> hp._Tensor:
        """Tensor containing the array of histograms"""
        return self._hist

    @property
    def nBins(self) -> int:
        """Number of bins in the histogram"""
        return self._nBins

    @property
    def nHist(self) -> int:
        """Number of histograms the integrator produces"""
        return self._nHist

    @property
    def nPhotons(self) -> int:
        """Number of photons in single ray hit"""
        return self._nPhotons

    def bindParams(self, **kwargs) -> None:
        """Binds extra params user provided shader code may have defined"""
        self._program.bindParams(**kwargs)

    def integrate(
        self, queue: hp._Tensor, t0: float, binSize: float, detectorId: int
    ) -> hp.Command:
        """
        Returns a command for processing the given queue.

        Parameters
        ----------
        queue: hp._Tensor
            Device address of the queue containing the samples to be processed
        t0: float
            Time point of the first bin edge
        binSize: float
            size of each bin
        detectorId: int
            Id of the detector. May be used by the response function.

        Returns
        -------
        cmd: hp.Command
            Command for dispatching the program to integrate the given queue
        """
        # create push constant
        push = self.Push(
            responseQueue=packUint64(queue.address),
            t0=t0,
            binSize=binSize,
            detectorId=detectorId,
        )
        # return dispatch command
        return self._program.dispatchPush(bytes(push), self.nHist)


class HistogramReducer:
    """
    Program for reducing local histograms into a single one.

    Parameters
    ----------
    nBins: int, default=256
        number of bins in the histogram
    code: Optional[bytes], default=None
        cached compiled code. Must match the configuration and might be
        different depending on the local machine.
        If None, the code get's compiled from source.
    """

    class Push(Structure):
        """Structure matching the push constant"""

        _fields_ = [
            ("histIn", uvec2),
            ("histOut", uvec2),
            ("normalization", c_float),
            ("nHist", c_uint32),
        ]

    def __init__(self, nBins: int = 256, *, code: Optional[bytes] = None) -> None:
        self._nBins = nBins

        subSize = hp.getSubgroupProperties().subgroupSize
        self._n = -(nBins // -subSize)  # work group size
        # compile code if necessary
        if code is None:
            # create preamble
            preamble = ""
            preamble += f"#define LOCAL_SIZE {subSize}\n"
            preamble += f"#define N_BINS {nBins}\n\n"
            # compile shader
            self._code = compileShader("wavefront.response.reduce.glsl", preamble)
        else:
            self._code = code
        # create program
        self._program = hp.Program(self._code)

    @property
    def nBins(self) -> int:
        """Number of bins in the histogram"""
        return self._nBins

    @property
    def code(self) -> bytes:
        """Compiled source code for given configuration"""
        return self._code

    def reduce(
        self, histOut: hp._Tensor, histIn: hp._Tensor, nHist: int, normalization: float
    ) -> hp.Command:
        """
        Returns a command for reducing the given histograms into a single one.
        If the output histogram already contains data, the new one will be
        added on top.

        Parameters
        ----------
        histOut: hp._Tensor
            tensor containing the output histogram: float[nBins]
        histIn: hp._Tensor
            tensor containing the input histograms: float[nBins][nHist]
        nHist: int
            number of input histograms
        normalization: float
            factor each bin should be multiplied with

        Returns
        -------
        cmd: hp.Command
            command for dispatching the reduce program with given params
        """
        # create push
        push = self.Push(
            histIn=packUint64(histIn.address),
            histOut=packUint64(histOut.address),
            nHist=nHist,
            normalization=normalization,
        )
        # dispatch program
        return self._program.dispatchPush(bytes(push), self._n)


class HostIntegrator:
    """
    Proxy integrator copying back samples from the GPU to the CPU without
    performing any calculations.

    Parameters
    ----------
    capacity: int
        Maximum number of samples the integrator can process in a single run.
    nPhotons: int
        Number of photons in single ray hit
    """

    def __init__(self, capacity: int, nPhotons: int) -> None:
        # save params
        self._capacity = capacity
        self._nPhotons = nPhotons

        # build sample type
        class PhotonHit(Structure):
            _fields_ = [
                ("wavelength", c_float),
                ("travelTime", c_float),
                ("log_radiance", c_float),
                ("throughput", c_float),
            ]

        class RayHit(Structure):
            _fields_ = [
                ("position", vec3),
                ("direction", vec3),
                ("normal", vec3),
                ("hits", PhotonHit * nPhotons),
            ]

        # calculate queue size
        photonSize = 16
        raySize = 36
        itemSize = raySize + nPhotons * photonSize
        totalSize = 4 + itemSize * capacity  # starts with int count
        sampleFloats = 9 + 4 * nPhotons
        totalFloats = sampleFloats * capacity
        # allocate memory
        self._buffer = hp.RawBuffer(totalSize)
        # pointer to counts
        adr = self._buffer.address
        self._count = c_uint32.from_address(adr)
        self._sampleType = RayHit * capacity
        self._structured = as_array((self._sampleType).from_address(adr + 4))
        self._flat = as_array((c_float * totalFloats).from_address(adr + 4)).reshape(
            (-1, sampleFloats)
        )

    @property
    def capacity(self) -> int:
        """Maximum number of samples the integrator can process in a single run"""
        return self._capacity

    @property
    def nPhotons(self) -> int:
        """Number of photons in single ray hit"""
        return self._nPhotons

    @property
    def sampleType(self) -> Type[Structure]:
        """Returns the structure describing a single sample"""
        return self._sampleType

    def numpy(self, structured: bool = True) -> NDArray:
        """
        Returns the numpy array containing the last retrieved samples.
        If structured is True, returns a structured numpy array, else a 2D float
        matrix with one row per sample. Each row has the following layout:

        | position | direction | normal |
            [ wavelength | travelTime | log_radiance | throughput ] * nPhotons
        """
        if structured:
            return self._structured[: self._count.value]
        else:
            return self._flat[: self._count.value]

    def integrate(self, queue: hp._Tensor) -> hp.Command:
        """
        Returns a command for copying the samples back to the CPU.

        Parameters
        ----------
        queue: int
            tensor containing the queue
        """
        return hp.retrieveTensor(queue, self._buffer)
