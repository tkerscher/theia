from __future__ import annotations

import hephaistos as hp
from hephaistos.queue import QueueBuffer, QueueTensor, QueueView, clearQueue
from hephaistos.pipeline import PipelineStage

from theia.items import createHitQueueItem
from theia.util import compileShader, loadShader

from ctypes import Structure, c_float, c_uint32
from typing import Callable, Dict, List, Optional, Type, Union


class HistogramReducer(PipelineStage):
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

    name = "Histogram Reducer"

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
        nHist: int = 1,
        nBins: int = 256,
        normalization: float = 1.0,
        blockSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        super().__init__({"Params": self.Params})

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
        if histIn is not None:
            self.histIn = histIn
        if histOut is not None:
            self.histOut = histOut
        self.setParams(norm=normalization, nHist=nHist)

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
    def code(self) -> bytes:
        """Compiled source code for given configuration"""
        return self._code

    # pipeline stage api

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        return [self._program.dispatch(self._groupSize)]


class HistogramEstimator(PipelineStage):
    """
    Estimator producing a light curve by computing the histogram of samples.

    Parameters
    ----------
    maxSamples: int
        Maximum number of samples the estimator can process in a single run.
    queue: hp.Tensor
        Tensor containing the hit queue to be processed
    code: str | Dict[str, bytes] | None
        Either the source code for the response function or the dict of
        compiled shaders. If None, the default Lambertian response function is
        used (black body). The compiled shader must match the configuration and
        might differ on different machines.
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
    clearQueue: bool, default=True
        True, if the queue should be reset/cleared after the estimator processed
        it.
    blockSize: int, default=128
        number of threads in a single local work group (block size in CUDA)
    """

    name = "Histogram Estimator"

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
        nPhotons: int = 4,
        t0: float = 0.0,
        binSize: float = 1.0,
        normalization: float = 1.0,
        clearQueue: bool = True,
        blockSize: int = 128,
    ) -> None:
        super().__init__({"Parameters": self.Params}, {"norm"})
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
            preamble += f"#define HIT_QUEUE_SIZE {maxSamples}\n"
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
            nBins=nBins,
            normalization=normalization,
            blockSize=blockSize,
            code=code["reducer"] if "reducer" in code else None,
        )
        code["reducer"] = self._reducer.code
        self._code = code
        # create program
        self._program = hp.Program(code["hist"])
        self._program.bindParams(Histograms=self._gpuHist, HitQueueBuffer=queue)

        # save params
        self._maxSamples = maxSamples
        self._nBins = nBins
        self._nPhotons = nPhotons
        self._queue = queue
        self._clearQueue = clearQueue
        self.setParams(binSize=binSize, detectorId=detectorId, t0=t0)

    @property
    def code(self) -> Dict[str, bytes]:
        """Compiled source code for given configuration"""
        return self._code

    @property
    def norm(self) -> float:
        """common factor each bin gets multiplied with"""
        return self._reducer.getParam("norm")

    @norm.setter
    def norm(self, value: float) -> None:
        self._reducer.setParam("norm", value)

    @property
    def clearQueue(self) -> bool:
        """
        True, if the queue should be reset/cleared after the estimator processed
        it.
        """
        return self._clearQueue

    @clearQueue.setter
    def clearQueue(self, value: bool) -> None:
        self._clearQueue = value

    @property
    def queue(self) -> hp.Tensor:
        """Queue containing the samples"""
        return self._queue

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
        self._bindParams(self._program, i)
        return [
            hp.clearTensor(self._reducedHist),
            self._program.dispatch(self._nHist),
            *([clearQueue(self.queue)] if self.clearQueue else []),
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
    clearQueue: bool, default=True
        True, if the queue should be reset/cleared after the estimator processed
        it.
    """

    name = "Host Estimator"

    def __init__(
        self, capacity: int, nPhotons: int, queue: hp.Tensor, *, clearQueue: bool = True
    ) -> None:
        super().__init__({})
        # save params
        self._capacity = capacity
        self._nPhotons = nPhotons
        self._queue = queue
        self._clearQueue = clearQueue

        # create queue
        self._item = createHitQueueItem(nPhotons)
        self._buffers = [QueueBuffer(self._item, capacity) for _ in range(2)]

    @property
    def capacity(self) -> int:
        """Maximum number of samples the estimator can process in a single run"""
        return self._capacity

    @property
    def item(self) -> Type[Structure]:
        """Structure describing a single item in the queue"""
        return self._item

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

    @property
    def clearQueue(self) -> bool:
        """
        True, if the queue should be reset/cleared after the estimator processed
        it.
        """
        return self._clearQueue

    @clearQueue.setter
    def clearQueue(self, value: bool) -> None:
        self._clearQueue = value

    def view(self, i: int) -> QueueView:
        """Returns the view into the i-th queue buffer"""
        return self._buffers[i].view

    # pipeline stage api

    def run(self, i: int) -> List[hp.Command]:
        return [
            hp.retrieveTensor(self.queue, self._buffers[i]),
            *([clearQueue(self.queue)] if self.clearQueue else []),
        ]


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
    """

    name = "Cached Sample Source"

    def __init__(
        self,
        capacity: int,
        nPhotons: int,
        updateFn: Optional[Callable[[CachedSampleSource, int], None]] = None,
    ) -> None:
        super().__init__({})
        # save params
        self._capacity = capacity
        self._nPhotons = nPhotons
        self._updateFn = updateFn

        # create queue
        self._item = createHitQueueItem(nPhotons)
        self._buffers = [QueueBuffer(self._item, capacity) for _ in range(2)]
        self._tensor = QueueTensor(self._item, capacity)

        # by default, copy whole buffer
        for i in range(2):
            self.view(i).count = capacity

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

    def view(self, i: int) -> QueueView:
        """Returns the view into the i-th queue"""
        return self._buffers[i].view

    # pipeline stage api

    def update(self, i: int) -> None:
        if self._updateFn is not None:
            self._updateFn(self, i)

    def run(self, i: int) -> List[hp.Command]:
        return [hp.updateTensor(self._buffers[i], self._tensor)]
