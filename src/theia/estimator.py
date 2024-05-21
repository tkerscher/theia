from __future__ import annotations
from abc import abstractmethod

import hephaistos as hp
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import QueueBuffer, QueueTensor, QueueView, clearQueue

from ctypes import Structure, c_float, c_uint32

from theia.util import ShaderLoader, compileShader, createPreamble

from numpy.typing import NDArray
from typing import Dict, List, Set, Type, Union, Optional


__all__ = [
    "createValueQueue",
    "CustomHitResponse",
    "EmptyResponse",
    "Estimator",
    "HistogramEstimator",
    "HistogramReducer",
    "HitItem",
    "HitRecorder",
    "HitReplay",
    "HitResponse",
    "PolarizedHitItem",
    "UniformHitResponse",
    "ValueHitResponse",
    "ValueItem",
]


def __dir__():
    return __all__


class HitItem(Structure):
    """
    Structure describing the layout of a single hit. Containing the rays
    direction, the position, surface normal and time of the hit on the detector
    surface, as well as the light's wavelength and MC contribution value.
    """

    _fields_ = [
        ("position", c_float * 3),
        ("direction", c_float * 3),
        ("normal", c_float * 3),
        ("wavelength", c_float),
        ("time", c_float),
        ("contrib", c_float),
    ]


class PolarizedHitItem(Structure):
    """
    Structure describing the layout of a single hit similar to `HitItem`, but
    with additional fields for polarization.  Polarization is given by a Stokes'
    vector and corresponding reference frame defined by the direction of
    vertical polarization of unit length and perpendicular to propagation
    direction.
    """

    _fields_ = [
        ("position", c_float * 3),
        ("direction", c_float * 3),
        ("normal", c_float * 3),
        ("stokes", c_float * 4),
        ("polarizationRef", c_float * 3),
        ("wavelength", c_float),
        ("time", c_float),
        ("contrib", c_float),
    ]


class HitResponse(SourceCodeMixin):
    """
    Base class for response functions called for each hit produced by a tracer.
    A response function is a GLSL function consuming a single `HitItem` as
    described in `createHitItem`:

    ```
    void response(HitItem item)
    ```
    """

    name = "Hit Response"

    def __init__(
        self, params: Dict[str, Type[Structure]] = {}, extra: Set[str] = set()
    ) -> None:
        super().__init__(params, extra)

    def prepare(self, maxHits: int, polarized: bool) -> None:
        """
        Called by e.g. a `Tracer` when binding the response into its program.
        Tells the response function how many hits it has to expect at most per
        run and can be used to make appropriate allocations.
        """
        pass


class EmptyResponse(HitResponse):
    """Empty hit response ignoring all hits"""

    def __init__(self) -> None:
        super().__init__()

    # sourceCode via descriptor
    sourceCode = ShaderLoader("response.empty.glsl")


class HitRecorder(HitResponse):
    """
    Records hits onto a queue, which can be retrieved later on.
    Should be placed in the pipeline at a later stage than the tracing.

    Parameters
    ----------
    polarized: bool, default=False
        Whether to save the polarization state.
    retrieve: bool, default=True
        Whether the queue gets retrieved from the device after processing.
        If `True`, clears the queue afterwards.

    Note
    ----
    If the hit source (e.g. a tracer) does not produce polarized samples, the
    polarization reference will be the null vector.
    """

    _sourceCode = ShaderLoader("response.record.glsl")

    def __init__(self, *, polarized: bool = False, retrieve: bool = True) -> None:
        super().__init__()
        # save params
        self._capacity = 0
        self._polarized = polarized
        self._retrieve = retrieve
        # set tensor and queue
        self._tensor = None
        self._buffer = [None for _ in range(2)]

    def prepare(self, maxHits: int, polarized: bool) -> None:
        self._capacity = maxHits
        # create queue
        item = PolarizedHitItem if polarized else HitItem
        self._tensor = QueueTensor(item, maxHits)
        hp.execute(clearQueue(self._tensor))
        if self.retrieve:
            self._buffer = [QueueBuffer(item, maxHits) for _ in range(2)]

    @property
    def capacity(self) -> int:
        """Maximum number of hits that can be saved per run"""
        return self._capacity

    @property
    def polarized(self) -> bool:
        """Whether to save the polarization state"""
        return self._polarized

    @property
    def retrieve(self) -> bool:
        """Wether the queue gets retrieved from the device after processing"""
        return self._retrieve

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            HIT_QUEUE_SIZE=self.capacity,
            HIT_QUEUE_POLARIZED=self.polarized,
        )
        return preamble + self._sourceCode

    @property
    def tensor(self) -> Union[QueueTensor, None]:
        """Tensor holding the queue storing the hits"""
        return self._tensor

    def buffer(self, i: int) -> Optional[QueueBuffer]:
        """Buffer holding the i-th queue. `None` if retrieve was set to `False`"""
        return self._buffer[i]

    def view(self, i: int) -> Optional[QueueView]:
        """View into the i-th queue. `None` if retrieve was set to `False`"""
        return self.buffer(i).view if self.retrieve else None

    def bindParams(self, program: hp.Program, i: int) -> None:
        if self.tensor is None:
            raise RuntimeError("Hit response has not been prepared")
        super().bindParams(program, i)
        program.bindParams(HitQueueOut=self._tensor)

    def run(self, i: int) -> List[hp.Command]:
        if self.retrieve:
            return [
                hp.retrieveTensor(self.tensor, self.buffer(i)),
                clearQueue(self.tensor),
            ]
        else:
            return []


class HitReplay(PipelineStage):
    """
    Reads hits from a queue and feeds each to the given response function.

    Parameters
    ----------
    capacity: int
        Maximum number of hits that can be processed per run
    response: HitResponse
        Response to be called on each hit
    polarized: bool, default=False
        Whether the hits contain polarization information.
    batchSize: int, default=128
        Number of hits processed per work group
    code: bytes | None, default=None
        Compiled source code. If `None`, the byte code get's compiled from
        source. Note, that the compiled code is not checked. You must ensure
        it matches the configuration.
    """

    name = "Hit Replay"

    def __init__(
        self,
        capacity: int,
        response: HitResponse,
        *,
        polarized: bool = False,
        batchSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        super().__init__()
        # save params
        self._batchSize = batchSize
        self._capacity = capacity
        self._polarized = polarized
        self._response = response

        # prepare response
        response.prepare(capacity, polarized)

        # create code if needed
        if code is None:
            preamble = createPreamble(
                BATCH_SIZE=batchSize,
                HIT_QUEUE_SIZE=capacity,
                HIT_QUEUE_POLARIZED=polarized,
                POLARIZATION=polarized,
            )
            headers = {"response.glsl": response.sourceCode}
            code = compileShader("response.replay.glsl", preamble, headers)
        self._code = code
        self._program = hp.Program(self._code)
        # calculate number of workgroups
        self._groups = -(capacity // -batchSize)

        # create queue
        item = PolarizedHitItem if polarized else HitItem
        self._tensor = QueueTensor(item, capacity)
        self._buffer = [QueueBuffer(item, capacity) for _ in range(2)]
        # set buffer item count to max by default
        for buffer in self._buffer:
            buffer.view.count = capacity
        # bind memory
        self._program.bindParams(HitQueueIn=self._tensor)

    @property
    def batchSize(self) -> int:
        """Number of hits processed per work group"""
        return self._batchSize

    @property
    def capacity(self) -> int:
        """Maximum number of hits that can be processed per run"""
        return self._capacity

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching"""
        return self._code

    @property
    def polarized(self) -> bool:
        """Whether the hits contain polarization information"""
        return self._polarized

    @property
    def response(self) -> HitResponse:
        """Response function called for each hit"""
        return self._response

    def buffer(self, i: int) -> QueueBuffer:
        """Buffer holding the i-th queue used in the next run"""
        return self._buffer[i]

    def view(self, i: int) -> QueueView:
        """View into the i-th queue holding the hits for the next run"""
        return self.buffer(i).view

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        return [
            hp.updateTensor(self.buffer(i), self._tensor),
            self._program.dispatch(self._groups),
        ]


class ValueItem(Structure):
    """
    Structure describing the layout of a single value item consumed by
    estimators. Each item assigns a value to a timestamp.
    """

    _fields_ = [("value", c_float), ("time", c_float)]


def createValueQueue(capacity: int) -> QueueTensor:
    """
    Util function for creating a queue of `ValueItem` with enough space to
    hold an amount of capacity items.
    """
    queue = QueueTensor(ValueItem, capacity)
    hp.execute(clearQueue(queue))  # sets count to zero
    return queue


class ValueHitResponse(HitResponse):
    """
    Template class using a provided value function creating a single float
    for each hit and stores it with its timestamp on a queue for further
    processing. Expects the value function as GLSL code:

    ```
    float responseValue(HitItem item)
    ```

    Parameters
    ----------
    queue: QueueTensor | None
        Queue in which the `ValueItem` will be stored.
        If `None` creates one during preparation.
    params: {str: Structure}, default={}
        Dictionary of named ctype structure containing the stage's parameters.
        Each structure will be allocated on the CPU side and twice on the GPU
        for buffering. The latter can be bound in programs
    extra: {str}, default={}
        Set of extra parameter name, that can be set and retrieved using the
        stage api. Take precedence over parameters defined by structs. Should
        be be implemented in subclasses as properties.
    """

    _templateCode = ShaderLoader("response.value.glsl")

    def __init__(
        self,
        queue: Union[QueueTensor, None],
        params: Dict[str, Type[Structure]] = {},
        extra: Set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self._queue = queue

    def prepare(self, maxHits: int, polarized: bool) -> None:
        if self._queue is None:
            self._queue = createValueQueue(maxHits)
        elif self.queue.capacity < maxHits:
            raise RuntimeError(
                f"Queue not big enough to store are hits: "
                f"queue has a capacity of {self.queue.capacity} but {maxHits} needed"
            )

    @property
    def queue(self) -> Union[QueueTensor, None]:
        """Tensor to be filled with `ValueItem` by this response function"""
        return self._queue

    @property
    @abstractmethod
    def valueFunction(self) -> str:
        """Source code of the value function"""
        pass

    @property
    def sourceCode(self) -> str:
        if self.queue is None:
            raise RuntimeError("Hit response has not been prepared")
        # create preamble
        preamble = f"#define VALUE_QUEUE_SIZE {self.queue.capacity}\n"
        # assemble full source code
        return "\n".join([preamble, self.valueFunction, self._templateCode])

    def bindParams(self, program: hp.Program, i: int) -> None:
        if self.queue is None:
            raise RuntimeError("Hit response has not been prepared")
        super().bindParams(program, i)
        program.bindParams(ValueQueueOut=self.queue)


class UniformHitResponse(ValueHitResponse):
    """
    Response function simulating a perfect isotropic and uniform response.

    Parameters
    ----------
    queue: QueueTensor | None, default=None
        Queue in which the `ValueItem` will be stored.
        If `None`, creates one during preparation.
    """

    def __init__(self, queue: Union[QueueTensor, None] = None) -> None:
        super().__init__(queue)

    # property via descriptor
    valueFunction = ShaderLoader("response.uniform.glsl")


class CustomHitResponse(ValueHitResponse):
    """
    Response function based on user provided code. Code must define the response
    value function:

    ```
    float responseValue(HitItem item)
    ```

    Parameters:
    -----------
    code: str
        User provided source code (GLSL).
    """

    def __init__(self, code: str, *, queue: Union[QueueTensor, None] = None) -> None:
        super().__init__(queue)
        # add include guards before
        _code = "#ifndef _INCLUDE_RESPONSE_CUSTOM\n"
        _code += "#define _INCLUDE_RESPONSE_CUSTOM\n"
        _code += code
        _code += "#endif\n"
        self._code = _code

    @property
    def valueFunction(self) -> str:
        return self._code


class Estimator(PipelineStage):
    """
    Base class for estimators that produce a final output by consuming a queue
    of `ValueItem` each consisting of a timestamp and a single float.

    Parameters
    ----------
    queue: QueueTensor
        Queue from which to consume the `ValueItem`
    clearQueue: bool
        Wether the input queue should be cleared after processing it
    params: {str: Structure}, default={}
        Dictionary of named ctype structure containing the stage's parameters.
        Each structure will be allocated on the CPU side and twice on the GPU
        for buffering. The latter can be bound in programs
    extra: {str}, default={}
        Set of extra parameter name, that can be set and retrieved using the
        stage api. Take precedence over parameters defined by structs. Should
        be be implemented in subclasses as properties.
    """

    name = "Estimator"

    def __init__(
        self,
        queue: QueueTensor,
        clearQueue: bool,
        params: Dict[str, Type[Structure]] = {},
        extra: Set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self._queue = queue
        self._clearQueue = clearQueue

    @property
    def clearQueue(self) -> bool:
        """Wether the input queue should be cleared after processing it"""
        return self._clearQueue

    @property
    def queue(self) -> QueueTensor:
        """Queue holding the `ValueItem` to be consumed by the estimator"""
        return self._queue


class HistogramReducer(PipelineStage):
    """
    Util class for reducing a set of histograms into a single one.

    Parameters
    ----------
    nBins: int
        Number of bins in the histogram
    nHist: int
        Number of histograms to reduce
    histIn: Tensor | None, default=None
        Tensor holding the histograms to reduce. If `None` creates a new one
        with enough space to hold `nHist` histograms.
    histOut: Tensor | None, default=None
        Tensor holding the final reduced histogram. If `None` creates a new one
        with enough space to hold the resulting histogram.
    normalization: float, default=1.0
        Common factor each bin gets multiplied with
    retrieve: bool, default=True
        Wether to retrieve the final histogram
    batchSize: int, default=128
        Number of threads per workgroup

    Stage Parameters
    ----------------
    normalization: float
        Common factor each bin gets multiplied with
    nHist: int
        Number of histograms to reduce
    """

    name = "Histogram Reducer"

    class Params(Structure):
        """Parameters"""

        _fields_ = [("normalization", c_float), ("nHist", c_uint32)]

    class Constants(Structure):
        """Specialization Constants"""

        _fields_ = [("batchSize", c_uint32), ("nBins", c_uint32)]

    # lazily compile code (shared among all instances, variation via spec consts)
    byte_code = None

    def __init__(
        self,
        *,
        nBins: int,
        nHist: int,
        histIn: Optional[hp.Tensor] = None,
        histOut: Optional[hp.Tensor] = None,
        retrieve: bool = True,
        normalization: float = 1.0,
        batchSize: int = 128,
    ) -> None:
        super().__init__({"Params": self.Params})
        # save params
        self._batchSize = batchSize
        self._nBins = nBins
        self._retrieve = retrieve
        self.setParams(normalization=normalization, nHist=nHist)

        # create code if needed
        if HistogramReducer.byte_code is None:
            HistogramReducer.byte_code = compileShader("estimator.reduce.glsl")
        # create specialization
        spec = HistogramReducer.Constants(batchSize=batchSize, nBins=nBins)
        # compile
        self._program = hp.Program(HistogramReducer.byte_code, bytes(spec))
        self._groups = -(nBins // -batchSize)

        # allocate memory if needed
        n = nBins * nHist
        self._histIn = hp.FloatTensor(n) if histIn is None else histIn
        self._histOut = hp.FloatTensor(nBins) if histOut is None else histOut
        self._buffer = [hp.FloatBuffer(nBins) if retrieve else None for _ in range(2)]
        # bind memory
        self._program.bindParams(HistogramIn=self._histIn, HistogramOut=self._histOut)

    @property
    def batchSize(self) -> int:
        """Number of threads per workgroup"""
        return self._batchSize

    @property
    def histIn(self) -> hp.Tensor:
        """Tensor containing the list of histograms to reduce"""
        return self._histIn

    @property
    def histOut(self) -> hp.Tensor:
        """Tensor containing the final reduced histogram"""
        return self._histOut

    @property
    def nBins(self) -> int:
        """Number of bins in the histogram"""
        return self._nBins

    @property
    def retrieve(self) -> bool:
        """Wether to retrieve the final histogram"""
        return self._retrieve

    def result(self, i: int) -> Optional[NDArray]:
        """The retrieved i-th histogram. `None` if `retrieve`was set to `False`"""
        return self._buffer[i].numpy() if self.retrieve else None

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        cmds = [self._program.dispatch(self._groups)]
        if self.retrieve:
            cmds.append(hp.retrieveTensor(self.histOut, self._buffer[i]))
        return cmds


class HistogramEstimator(Estimator):
    """
    Estimator producing a histogram from the provided value samples as a
    function of time in a given frame.

    Parameters
    ----------
    queue: QueueTensor
        Queue from which to consume the `ValueItem`
    nBins: int, default=100
        Number of bins in the histogram
    t0: float, default=0.0
        First bin edge, i.e. the minimum time value a sample has to have to get
        included
    binSize: float, default=1.0
        Size of a single bin in unit of time
    normalization: float, default=1.0
        Common factor each bin gets multiplied with
    clearQueue: bool, default=True
        Wether the input queue should be cleared after processing it
    retrieve: bool, default=True
        Wether to retrieve the final histogram
    batchSize: int, default=128
        Number of items processed per workgroup during item processing
    batchSizeReduce: int, default=128
        Number of items processed per workgroup during reduction
    code: bytes | None, default=None
        Compiled source code. If `None`, the byte code get's compiled from
        source. Note, that the compiled code is not checked. You must ensure
        it matches the configuration.

    Stage Parameters
    ----------------
    t0: float
        First bin edge, i.e. the minimum time value a sample has to have to get
        included
    binSize: float, default=1.0
        Size of a single bin in unit of time
    normalization
        Common factor each bin gets multiplied with
    """

    class Params(Structure):
        """Histogram Params"""

        _fields_ = [
            ("t0", c_float),
            ("binSize", c_float),
        ]

    def __init__(
        self,
        queue: QueueTensor,
        *,
        nBins: int = 100,
        t0: float = 0.0,
        binSize: float = 1.0,
        normalization: float = 1.0,
        clearQueue: bool = True,
        retrieve: bool = True,
        batchSize: int = 128,
        batchSizeReduce: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        super().__init__(
            queue, clearQueue, {"Parameters": self.Params}, {"normalization"}
        )
        # save params
        self._batchSize = batchSize
        self.setParams(t0=t0, binSize=binSize)

        # create reducer
        self._groups = -(queue.capacity // -batchSize)
        self._reducer = HistogramReducer(
            nBins=nBins,
            nHist=self._groups,
            normalization=normalization,
            retrieve=retrieve,
            batchSize=batchSizeReduce,
        )

        # compile code if needed
        if code is None:
            preamble = createPreamble(
                BATCH_SIZE=batchSize,
                N_BINS=nBins,
                VALUE_QUEUE_SIZE=queue.capacity,
            )
            code = compileShader("estimator.hist.glsl", preamble)
        self._code = code
        # create program
        self._program = hp.Program(code)
        # bind memory
        self._program.bindParams(HistogramOut=self._reducer.histIn, ValueIn=self.queue)

    @property
    def batchSize(self) -> int:
        """Number of threads per workgroup during item processing"""
        return self._batchSize

    @property
    def batchSizeReduce(self) -> int:
        """Number of threads per workgroup during reduction"""
        return self._reducer.batchSize

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching"""
        return self._code

    @property
    def nBins(self) -> int:
        """Number of bins in the histogram"""
        return self._reducer.nBins

    @property
    def normalization(self) -> float:
        """Common factor each bin gets multiplied with"""
        return self._reducer.getParam("normalization")

    @normalization.setter
    def normalization(self, value: float) -> None:
        self._reducer.setParam("normalization", value)

    @property
    def resultTensor(self) -> hp.Tensor:
        """Tensor holding the final histogram"""
        return self._reducer.histOut

    @property
    def retrieve(self) -> bool:
        """Wether to retrieve the final histogram"""
        return self._reducer.retrieve

    def result(self, i: int) -> Optional[NDArray]:
        """The retrieved i-th histogram. `None` if `retrieve`was set to `False`"""
        return self._reducer.result(i)

    # Pipeline API

    def update(self, i: int) -> None:
        # also update reducer
        self._reducer.update(i)
        super().update(i)

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        return [
            self._program.dispatch(self._groups),
            *([clearQueue(self.queue)] if self.clearQueue else []),
            hp.flushMemory(),
            *self._reducer.run(i),
        ]


class HostEstimator(Estimator):
    """
    Copies `ValueItems` back to host without processing them.

    Parameters
    ----------
    queue: QueueTensor
        Queue from which to consume the `ValueItem`
    clearQueue: bool, default=True
        Wether the input queue should be cleared after processing it
    """

    def __init__(self, queue: QueueTensor, *, clearQueue: bool = True) -> None:
        super().__init__(queue, clearQueue)
        # create local copy
        self._buffers = [QueueBuffer(ValueItem, queue.capacity) for _ in range(2)]

    def buffer(self, i: int) -> QueueBuffer:
        """Returns the i-th queue buffer."""
        return self._buffers[i]

    def view(self, i: int) -> QueueView:
        """Returns a view of the i-th queue buffer"""
        return self.buffer(i).view

    def run(self, i: int) -> List[hp.Command]:
        return [
            hp.retrieveTensor(self.queue, self.buffer(i)),
            *([clearQueue(self.queue)] if self.clearQueue else []),
        ]
