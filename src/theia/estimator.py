import hephaistos as hp

from abc import ABC, abstractmethod
from ctypes import Structure, c_float, c_uint32, sizeof
from hephaistos.glsl import uvec2, vec3
from numpy.ctypeslib import as_array
from numpy.typing import NDArray
from typing import List, Optional, Type, Union
from .util import compileShader, loadShader, packUint64


class Estimator(ABC):
    """Base class for estimators"""

    @abstractmethod
    def estimate(self, queue: hp.Tensor, detectorId: int) -> List[hp.Command]:
        """
        Returns a list commands to be executed in sequence to create an estimate
        using the samples stored in the provided queue for the given detector.

        Parameters
        ----------
        queue: Union[hp.Tensor, int]
            Either its address or the tensor itself containing the queue holding
            the samples to be used for the estimate
        detectorId: int
            Id of the detector of which to create the estimate
        """
        pass


class HistogramReducer:
    """
    Helper program for reducing local histograms into a single one.

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
        self,
        histOut: Union[hp.Tensor, int],
        histIn: Union[hp.Tensor, int],
        nHist: int,
        normalization: float,
    ) -> hp.Command:
        """
        Returns a command for reducing the given histograms into a single one.
        If the output histogram already contains data, the new one will be
        added on top.

        Parameters
        ----------
        histOut: hp.Tensor | int
            either the address or the tensor itself containing the output
            histogram: float[nBins]
        histIn: hp.Tensor | int
            either the address or the tensor itself containing the input
            histograms: float[nBins][nHist]
        nHist: int
            number of input histograms
        normalization: float
            factor each bin should be multiplied with

        Returns
        -------
        cmd: hp.Command
            command for dispatching the reduce program with given params
        """
        # retrieve addresses if needed
        if isinstance(histIn, hp.Tensor):
            histIn = histIn.address
        if isinstance(histOut, hp.Tensor):
            histOut = histOut.address

        # create push
        push = self.Push(
            histIn=packUint64(histIn),
            histOut=packUint64(histOut),
            nHist=nHist,
            normalization=normalization,
        )
        # dispatch program
        return self._program.dispatchPush(bytes(push), self._n)


class HistogramEstimator(Estimator):
    """
    Estimator producing a light curve by computing the histogram of samples.

    Parameters
    ----------
    code: str | bytes | None
        Either the source code for the response function or the compiled shader.
        If None, the default Lambertian response function is used (black body).
        The compiled shader must match the configuration and might differ on
        different machines.
    maxSamples: int
        Maximum number of samples the estimator can process in a single run.
    nDetectors: int, default=0
        Number of detectors. Used for allocating separate histograms for each
        detector. If 0, only a single histogram is used for all estimates
    nBins: int, default=256
        Number of bins in the histogram
    nPhotons: int, default=4
        Number of photons in single ray hit
    t0: float, default=0.0
        first bin edge, i.e. earliest time a sample gets binned
    binSize: float, default=1.0
        size of a single bin (unit of time)
    normalization: float, default=1.0
        common factor each bin gets multiplied with
    batchSize: int, default=128
        Number of samples processed in a single batch
    reducerCode: bytes|None, default=None
        Compiled code of the internal histogram reducer instance. If None, gets
        compiled from source. The compiled shader must match the configuration
        and might differ on different machines.
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
        code: Union[str, bytes, None],
        maxSamples: int,
        *,
        nDetectors: int = 0,
        nBins: int = 256,
        nPhotons: int = 4,
        t0: float = 0.0,
        binSize: float = 1.0,
        normalization: float = 1.0,
        batchSize: int = 128,
        reducerCode: Optional[bytes] = None,
    ) -> None:
        # save params
        self._maxSamples = maxSamples
        self._nDetectors = nDetectors
        self._nBins = nBins
        self._nPhotons = nPhotons
        self._batchSize = batchSize
        self.binSize = binSize
        self.normalization = normalization
        self.t0 = t0

        # allocate local (GPU) histograms
        self._nHist = -(maxSamples // -batchSize)  # ceil division
        self._gpuHist = hp.FloatTensor(self._nHist * nBins)
        self._reducedHist = hp.FloatTensor(nBins)
        # allocate result (CPU) histograms
        # one big buffer -> use offset to retrieve individual ones
        if nDetectors == 0:
            nDetectors = 1
        self._cpuHist = hp.RawBuffer(4 * nBins * nDetectors)

        # compile source code if needed
        if code is None:
            code = loadShader("response.lambertian.glsl")
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
        self._program.bindParams(Histograms=self._gpuHist)

        # we'll also need a reducer
        self._reducer = HistogramReducer(nBins, code=reducerCode)

    @property
    def batchSize(self) -> int:
        """Number of samples processed in a single batch"""
        return self._batchSize

    @property
    def binSize(self) -> float:
        """size of a single bin (unit of time)"""
        return self._binSize

    @binSize.setter
    def binSize(self, value: float) -> None:
        self._binSize = value

    @property
    def capacity(self) -> int:
        """Maximum number of samples the estimator can process in a single run"""
        return self._capacity

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching compilation"""
        return self._code

    @property
    def histograms(self) -> hp.Tensor:
        """Tensor containing the array of histograms"""
        return self._hist

    @property
    def nBins(self) -> int:
        """Number of bins in the histogram"""
        return self._nBins

    @property
    def nHist(self) -> int:
        """Number of histograms the estimator produces"""
        return self._nHist

    @property
    def nPhotons(self) -> int:
        """Number of photons in single ray hit"""
        return self._nPhotons

    @property
    def normalization(self) -> float:
        """common factor each bin gets multiplied with"""
        return self._normalization

    @normalization.setter
    def normalization(self, value: float) -> None:
        self._normalization = value

    @property
    def reducerCode(self) -> bytes:
        """Compiled code of the internal histogram reducer instance"""
        return self._reducer.code

    @property
    def t0(self) -> float:
        """first bin edge, i.e. earliest time a sample gets binned"""
        return self._t0

    @t0.setter
    def t0(self, value: float) -> None:
        self._t0 = value

    def bindParams(self, **kwargs) -> None:
        """Binds extra params user provided shader code may have defined"""
        self._program.bindParams(**kwargs)

    def histogram(self, id: int = 0) -> NDArray:
        """
        Returns the histogram of the given detector specified via its id.
        Use id 0, if a single histogram is shared.
        """
        address = self._cpuHist.address
        address += 4 * id * self.nBins  # skip nBins floats per id
        array = (c_float * self.nBins).from_address(address)
        return as_array(array)

    def estimate(self, queue: hp.Tensor, detectorId: int) -> List[hp.Command]:
        """
        Returns a list commands to be executed in sequence to create an estimate
        using the samples stored in the provided queue for the given detector.

        Parameters
        ----------
        queue: hp.Tensor
            tensor containing the queue holding the samples to be used for the
            estimate
        detectorId: int
            Id of the detector of which to create the estimate
        """
        # create push constant
        push = self.Push(
            responseQueue=packUint64(queue.address),
            t0=self.t0,
            binSize=self.binSize,
            detectorId=detectorId,
        )

        # return sequence: histogram + reducer
        return [
            hp.clearTensor(self._reducedHist),
            self._program.dispatchPush(bytes(push), self.nHist),
            hp.flushMemory(),
            self._reducer.reduce(
                self._reducedHist, self._gpuHist, self.nHist, self.normalization
            ),
            hp.retrieveTensor(
                self._reducedHist,
                self._cpuHist,
                bufferOffset=4 * self.nBins * detectorId,  # bytes
                size=4 * self.nBins,
            ),
        ]


class HostEstimator(Estimator):
    """
    Proxy estimator copying back samples from the GPU to the CPU without
    performing any calculations.

    Parameters
    ----------
    capacity: int
        Maximum number of samples the estimator can process in a single run.
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
                ("time", c_float),
                ("contribution", c_float),
            ]

        class RayHit(Structure):
            _fields_ = [
                ("position", vec3),
                ("direction", vec3),
                ("normal", vec3),
                ("hits", PhotonHit * nPhotons),
            ]

        # calculate queue size
        itemSize = sizeof(RayHit)
        totalSize = 4 + itemSize * capacity  # starts with int count
        sampleFloats = 9 + 3 * nPhotons
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
        """Maximum number of samples the estimator can process in a single run"""
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

        | position | direction | normal | hits[nPhotons]:
            (wavelength | time | contribution)
        """
        if structured:
            return self._structured[: self._count.value]
        else:
            return self._flat[: self._count.value]

    def estimate(self, queue: hp.Tensor, detectorId: int) -> List[hp.Command]:
        """
        Returns a list commands to be executed in sequence to create an estimate
        using the samples stored in the provided queue for the given detector.

        Parameters
        ----------
        queue: hp.Tensor
            tensor containing the queue holding the samples to be used for the
            estimate
        detectorId: int
            Id of the detector of which to create the estimate
        """
        return [hp.retrieveTensor(queue, self._buffer)]


class CachedEstimator:
    """
    Utility class for running a different estimator using samples loaded from
    the CPU, e.g. cached samples from a previous run.

    Parameters
    ----------
    estimator: Estimator
        Estimator instance used for processing the cached samples
    capacity: int
        Maximum number of samples the estimator can process in a single run
    nPhotons: int
        Number of photons in single ray hit
    """

    def __init__(self, estimator: Estimator, capacity: int, nPhotons: int) -> None:
        # save params
        self._estimator = estimator
        self._capacity = capacity
        self._nPhotons = nPhotons

        # build sample type
        class PhotonHit(Structure):
            _fields_ = [
                ("wavelength", c_float),
                ("time", c_float),
                ("contribution", c_float),
            ]

        class RayHit(Structure):
            _fields_ = [
                ("position", vec3),
                ("direction", vec3),
                ("normal", vec3),
                ("hits", PhotonHit * nPhotons),
            ]

        # calculate queue size
        itemSize = sizeof(RayHit)
        totalSize = 4 + itemSize * capacity  # starts with int count
        sampleFloats = 9 + 3 * nPhotons
        totalFloats = sampleFloats * capacity
        # allocate memory
        self._buffer = hp.RawBuffer(totalSize)
        self._tensor = hp.ByteTensor(totalSize)
        # pointer to counts
        adr = self._buffer.address
        self._count = c_uint32.from_address(adr)
        self._sampleType = RayHit * capacity
        self._structured = as_array((self._sampleType).from_address(adr + 4))
        self._flat = as_array((c_float * totalFloats).from_address(adr + 4)).reshape(
            (-1, sampleFloats)
        )
        # set queue count to max so we dont need to update it
        self._count.value = capacity

    @property
    def estimator(self) -> Estimator:
        """Estimator instance used for processing the cached samples"""
        return self._estimator

    @property
    def capacity(self) -> int:
        """Maximum number of samples the estimator can process in a single run"""
        return self._capacity

    @property
    def nPhotons(self) -> int:
        """Number of photons in single ray hit"""
        return self._nPhotons

    def numpy(self, structured: bool = True) -> NDArray:
        """
        Returns the numpy array containing the last retrieved samples.
        If structured is True, returns a structured numpy array, else a 2D float
        matrix with one row per sample. Each row has the following layout:

        | position | direction | normal | hits[nPhotons]:
            (wavelength | time | contribution)
        """
        if structured:
            return self._structured
        else:
            return self._flat

    def estimateAsync(self, detectorId: int) -> List[hp.Command]:
        """
        Returns a list commands to be executed in sequence to create an estimate
        using the samples stored in the provided queue for the given detector.
        Check the estimator for the result.

        Parameters
        ----------
        detectorId: int
            Id of the detector of which to create the estimate
        """
        return [
            hp.updateTensor(self._buffer, self._tensor),
            *self.estimator.estimate(self._tensor, detectorId),
        ]

    def estimate(self, detectorId: int) -> None:
        """
        Runs the estimator and waits for it to finish.
        Check the estimator for the result.

        Parameters
        ----------
        detectorId: int
            Id of the detector of which to create the estimate
        """
        hp.executeList(self.estimateAsync(detectorId))
