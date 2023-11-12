from __future__ import annotations
import hephaistos as hp

from ctypes import Structure, c_float, c_uint32, sizeof
from hephaistos.glsl import vec3
from theia.scheduler import CachedPipelineStage, PipelineStage
from numpy.ctypeslib import as_array
from numpy.typing import NDArray
from typing import Callable, Dict, List, Optional, Type, Union
from .util import compileShader, loadShader


class HistogramReducer(CachedPipelineStage):
    """
    Helper program for reducing local histograms into a single one.

    Parameters
    ----------
    histIn: Optional[hp.Tensor], default=None
        Tensor containing the histograms to reduce.
        Must be set before sampling.
    histOut: Optional[hp.Tensor], default=None
        Tensor containing the histogram where the result will be stored.
        Must be set before sampling.
    nConfigs: int, default=2
        Number of pipeline configurations
    nHist: int, default=1
        Number of histograms to reduce
    nBins: int, default=256
        number of bins in the histogram
    normalization: float, default=1.0
        Normalization constant each bin gets multiplied with
    blockSize: int, default=128
        number of threads in a single local work grouop (block size in CUDA)
    code: Optional[bytes], default=None
        cached compiled code. Must match the configuration and might be
        different depending on the local machine.
        If None, the code get's compiled from source.
    """

    class Params(Structure):
        """Structure matching the parameter uniform"""

        _fields_ = [
            ("norm", c_float),
            ("nHist", c_uint32),
        ]

    def __init__(
        self,
        *,
        histIn: Optional[hp.Tensor] = None,
        histOut: Optional[hp.Tensor] = None,
        nConfigs: int = 2,
        nHist: int = 1,
        nBins: int = 256,
        normalization: float = 1.0,
        blockSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        super().__init__({"Params": self.Params}, nConfigs)

        # work group size
        self._groupSize = -(nBins // -blockSize)

        # compile code if necessary
        if code is None:
            # create preamble
            preamble = ""
            preamble += f"#define LOCAL_SIZE {blockSize}\n"
            preamble += f"#define N_BINS {nBins}\n\n"
            # compile shader
            self._code = compileShader("estimator.reduce.glsl", preamble)
        else:
            self._code = code
        # create program
        self._program = hp.Program(self._code)

        # save params
        self._nBins = nBins
        self.nHist = nHist
        self.normalization = normalization
        if histIn is not None:
            self.histIn = histIn
        if histOut is not None:
            self.histOut = histOut

    @property
    def histIn(self) -> hp.Tensor:
        """Tensor containing the histograms to reduce"""
        return self._histIn

    @histIn.setter
    def histIn(self, value: hp.Tensor) -> None:
        self._histIn = value
        self._program.bindParams(HistogramIn=value)

    @property
    def histOut(self) -> hp.Tensor:
        """Tensor containing the histogram where the result will be stored"""
        return self._histOut

    @histOut.setter
    def histOut(self, value: hp.Tensor) -> None:
        self._histOut = value
        self._program.bindParams(HistogramOut=value)

    @property
    def nBins(self) -> int:
        """Number of bins in the histogram"""
        return self._nBins

    @property
    def nHist(self) -> int:
        """Number of histograms to reduce"""
        return self._getParam("Params", "nHist")

    @nHist.setter
    def nHist(self, value: int) -> None:
        self._setParam("Params", "nHist", value)

    @property
    def normalization(self) -> float:
        """Normalization constant each bin gets multiplied with"""
        return self._getParam("Params", "norm")

    @normalization.setter
    def normalization(self, value: float) -> None:
        self._setParam("Params", "norm", value)

    @property
    def code(self) -> bytes:
        """Compiled source code for given configuration"""
        return self._code

    # pipeline stage api

    def run(self, i: int) -> List[hp.Command]:
        self._bindConfigs(self._program, i)
        return [self._program.dispatch(self._groupSize)]


class HistogramEstimator(CachedPipelineStage):
    """
    Estimator producing a light curve by computing the histogram of samples.

    Parameters
    ----------
    maxSamples: int
        Maximum number of samples the estimator can process in a single run.
    code: str | Dict[str, bytes] | None
        Either the source code for the response function or the dict of
        compiled shaders. If None, the default Lambertian response function is
        used (black body). The compiled shader must match the configuration and
        might differ on different machines.
    nBins: int, default=256
        Number of bins in the histogram
    nConfigs: int, default=2
        Number of pipeline configurations
    nPhotons: int, default=4
        Number of photons in single ray hit
    t0: float, default=0.0
        first bin edge, i.e. earliest time a sample gets binned
    binSize: float, default=1.0
        size of a single bin (unit of time)
    normalization: float, default=1.0
        common factor each bin gets multiplied with
    blockSize: int, default=128
        number of threads in a single local work group (block size in CUDA)
    """

    class Params(Structure):
        """Struct defining parameter uniform"""

        _fields_ = [
            ("t0", c_float),
            ("binSize", c_float),
            ("detectorId", c_uint32),
        ]

    def __init__(
        self,
        maxSamples: int,
        queue: hp.Tensor,
        code: Union[str, Dict[str, bytes], None] = None,
        *,
        detectorId: int = 0,
        nBins: int = 256,
        nConfigs: int = 2,
        nPhotons: int = 4,
        t0: float = 0.0,
        binSize: float = 1.0,
        normalization: float = 1.0,
        blockSize: int = 128,
    ) -> None:
        super().__init__({"Parameters": self.Params}, nConfigs)
        # allocate histograms
        self._nHist = -(maxSamples // -blockSize)  # ceil division
        self._gpuHist = hp.FloatTensor(self._nHist * nBins)
        self._reducedHist = hp.FloatTensor(nBins)

        # compile source code if needed
        if code is None:
            code = loadShader("response.lambertian.glsl")
        if type(code) is str:
            # create preamble
            preamble = ""
            preamble += f"#define BATCH_SIZE {blockSize}\n"
            preamble += f"#define N_BINS {nBins}\n"
            preamble += f"#define N_PHOTONS {nPhotons}\n\n"
            # compile source code
            headers = {"response.glsl": code}
            code = {"hist": compileShader("estimator.hist.glsl", preamble, headers)}

        # create reducer
        self._reducer = HistogramReducer(
            histIn=self._gpuHist,
            histOut=self._reducedHist,
            nHist=self._nHist,
            nConfigs=nConfigs,
            nBins=nBins,
            normalization=normalization,
            blockSize=blockSize,
            code=code["reducer"] if "reducer" in code else None,
        )
        code["reducer"] = self._reducer.code
        self._code = code
        # create program
        self._program = hp.Program(code["hist"])
        self._program.bindParams(Histograms=self._gpuHist, ResponseQueue=queue)

        # save params
        self._maxSamples = maxSamples
        self._nBins = nBins
        self._nPhotons = nPhotons
        self._queue = queue
        self.binSize = binSize
        self.detectorId = detectorId
        # self.normalization = normalization
        self.t0 = t0

    @property
    def code(self) -> Dict[str, bytes]:
        """Compiled source code for given configuration"""
        return self._code

    @property
    def binSize(self) -> float:
        """Width of a single bin"""
        return self._getParam("Parameters", "binSize")

    @binSize.setter
    def binSize(self, value: float) -> None:
        self._setParam("Parameters", "binSize", value)

    @property
    def detectorId(self) -> int:
        """
        Id of the detector the estimate is created for.
        May be used by the response function.
        """
        self._getParam("Parameters", "detectorId")

    @detectorId.setter
    def detectorId(self, value: int) -> None:
        self._setParam("Parameters", "detectorId", value)

    @property
    def normalization(self) -> float:
        """common factor each bin gets multiplied with"""
        return self._reducer.normalization

    @normalization.setter
    def normalization(self, value: float) -> None:
        self._reducer.normalization = value

    @property
    def queue(self) -> hp.Tensor:
        """Queue containing the samples"""
        return self._queue

    @property
    def t0(self) -> float:
        """First edge of the histogram"""
        return self._getParam("Parameters", "t0")

    @t0.setter
    def t0(self, value: float) -> None:
        self._setParam("Parameters", "t0", value)

    @property
    def histogram(self) -> hp.Tensor:
        """Tensor containing the result histogram"""
        return self._reducedHist

    @property
    def nBins(self) -> int:
        """Number of bins in the histogram"""
        return self._nBins

    @property
    def nPhotons(self) -> int:
        """Number of photons in single ray hit"""
        return self._nPhotons

    @property
    def maxSamples(self) -> int:
        """Maximum number of samples the estimator can process in a single run"""
        return self._maxSamples

    def bindParams(self, **kwargs) -> None:
        """Binds extra params user provided shader code may have defined"""
        self._program.bindParams(**kwargs)

    # pipeline stage api

    def update(self, i: int) -> None:
        self._reducer.update(i)
        super().update(i)

    def run(self, i: int) -> List[hp.Command]:
        self._bindConfigs(self._program, i)
        return [
            hp.clearTensor(self._reducedHist),
            self._program.dispatch(self._nHist),
            hp.flushMemory(),
            *self._reducer.run(i),
        ]


class HostEstimator(PipelineStage):
    """
    Proxy estimator copying back samples from the GPU to the CPU without
    performing any calculations.

    Parameters
    ----------
    capacity: int
        Maximum number of samples the estimator can process in a single run.
    nPhotons: int
        Number of photons in single ray hit
    queue: hp.Tensor
        Tensor containing the response queue from which to copy the samples.
    nBuffers: int, default=2
        Number of buffers
    """

    def __init__(
        self, capacity: int, nPhotons: int, queue: hp.Tensor, *, nBuffers: int = 2
    ) -> None:
        super().__init__(nBuffers)
        # save params
        self._capacity = capacity
        self._nPhotons = nPhotons
        self._queue = queue

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
        self._buffers = [hp.RawBuffer(totalSize) for _ in range(nBuffers)]
        # pointer to counts
        adrs = [b.address for b in self._buffers]
        self._counts = [c_uint32.from_address(adr) for adr in adrs]
        self._sampleType = RayHit * capacity
        self._structured = [
            as_array((self._sampleType).from_address(adr + 4)) for adr in adrs
        ]
        self._flat = [
            as_array((c_float * totalFloats).from_address(adr + 4)).reshape(
                (-1, sampleFloats)
            )
            for adr in adrs
        ]

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

    @property
    def queue(self) -> hp.Tensor:
        """
        Tensor containing the response queue from which to copy the samples.
        Must be set before running
        """
        return self._queue

    def numpy(self, i: int, structured: bool = True) -> NDArray:
        """
        Returns the numpy array using the memory of the i-th buffer containing
        the last retrieved samples.
        If structured is True, returns a structured numpy array, else a 2D float
        matrix with one row per sample. Each row has the following layout:

        | position | direction | normal | hits[nPhotons]:
            (wavelength | time | contribution)
        """
        if structured:
            return self._structured[i][: self._counts[i].value]
        else:
            return self._flat[i][: self._counts[i].value]

    # pipeline stage api

    def run(self, i: int) -> List[hp.Command]:
        return [hp.retrieveTensor(self.queue, self._buffers[i])]


class CachedSampleSource(PipelineStage):
    """
    Utility class for feeding cached samples (e.g. from HostEstimator) back into
    the pipeline.

    Parameters
    ----------
    capacity: int
        Maximum number of samples per run
    nPhotons: int
        Number of photons in single ray hit
    updateFn: (numpy array) -> None | None, default=None
        Optional function to be called before a task is run on the pipeline.
        Called with this object and the index of the buffer to fill.
        May be used to load data
    nBuffers: int, default=2
        Number of buffers
    """

    def __init__(
        self,
        capacity: int,
        nPhotons: int,
        updateFn: Optional[Callable[[CachedSampleSource, int], None]] = None,
        *,
        nBuffers: int = 2,
    ) -> None:
        super().__init__(nBuffers)
        # save params
        self._capacity = capacity
        self._nPhotons = nPhotons
        self._updateFn = updateFn

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
        self._tensor = hp.ByteTensor(totalSize)
        self._buffers = [hp.RawBuffer(totalSize) for _ in range(nBuffers)]
        # pointer to counts
        adrs = [b.address for b in self._buffers]
        self._counts = [c_uint32.from_address(adr) for adr in adrs]
        self._sampleType = RayHit * capacity
        self._structured = [
            as_array((self._sampleType).from_address(adr + 4)) for adr in adrs
        ]
        self._flat = [
            as_array((c_float * totalFloats).from_address(adr + 4)).reshape(
                (-1, sampleFloats)
            )
            for adr in adrs
        ]
        # by default, copy whole buffer
        for i in range(nBuffers):
            self.setCount(i, capacity)

    @property
    def capacity(self) -> int:
        """Maximum number of samples the estimator can process in a single run"""
        return self._capacity

    @property
    def nPhotons(self) -> int:
        """Number of photons in single ray hit"""
        return self._nPhotons

    @property
    def queue(self) -> hp.Tensor:
        """Tensor containing the cached samples on the GPU"""
        return self._tensor

    def setCount(self, i: int, count: int) -> int:
        """Set the number of samples in the i-th buffer."""
        self._counts[i].value = count

    def numpy(self, i: int, structured: bool = True) -> NDArray:
        """
        Returns the numpy array of the i-th buffer containing the last retrieved
        samples.
        If structured is True, returns a structured numpy array, else a 2D float
        matrix with one row per sample. Each row has the following layout:

        | position | direction | normal | hits[nPhotons]:
            (wavelength | time | contribution)
        """
        if structured:
            return self._structured[i]
        else:
            return self._flat[i]

    # pipeline stage api

    def update(self, i: int) -> None:
        if self._updateFn is not None:
            self._updateFn(self, i)

    def run(self, i: int) -> List[hp.Command]:
        return [hp.updateTensor(self._buffers[i], self._tensor)]
