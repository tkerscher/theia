from __future__ import annotations
from dataclasses import dataclass

import hephaistos as hp
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import QueueBuffer, QueueTensor, QueueView, clearQueue

from ctypes import Structure, c_float, c_uint32

from theia.util import ShaderLoader, compileShader, createPreamble
import theia.units as u

from numpy.typing import NDArray


__all__ = [
    "createValueQueue",
    "CustomValueResponse",
    "EmptyResponse",
    "Estimator",
    "HistogramEstimator",
    "HistogramReducer",
    "HitItem",
    "HitRecorder",
    "HitReplay",
    "HitResponse",
    "PolarizedHitItem",
    "StoreValueHitResponse",
    "UniformValueResponse",
    "ValueHitResponse",
    "ValueItem",
    "ValueResponse",
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


@dataclass
class TraceConfig:
    """
    Description of the tracing.

    Parameters
        ----------
        batchSize: int
            Total number of threads per run.
        blockSize: int
            Number of threads per work group.
        maxHits: int
            Maximum number of hits the response will be called per batch.
        normalization: float
            Normalization factor that must be applied to each sample to get a
            correct estimate.
        polarized: bool
            True, if hits contain polarization information.
    """

    batchSize: int
    blockSize: int
    maxHits: int
    normalization: float
    polarized: bool


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
        self, params: dict[str, type[Structure]] = {}, extra: set[str] = set()
    ) -> None:
        super().__init__(params, extra)

    def prepare(self, config: TraceConfig) -> None:
        """
        Called by e.g. a `Tracer` when binding the response into its program
        notifying it about its configuration to allow it to make adequate
        adjustments to itself beforehand.
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

    def prepare(self, config: TraceConfig) -> None:
        self._capacity = config.maxHits
        # create queue
        item = PolarizedHitItem if config.polarized else HitItem
        self._tensor = QueueTensor(item, config.maxHits)
        hp.execute(clearQueue(self._tensor))
        if self.retrieve:
            self._buffer = [QueueBuffer(item, config.maxHits) for _ in range(2)]

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
    def tensor(self) -> QueueTensor | None:
        """Tensor holding the queue storing the hits"""
        return self._tensor

    def buffer(self, i: int) -> QueueBuffer | None:
        """Buffer holding the i-th queue. `None` if retrieve was set to `False`"""
        return self._buffer[i]

    def view(self, i: int) -> QueueView | None:
        """View into the i-th queue. `None` if retrieve was set to `False`"""
        return self.buffer(i).view if self.retrieve else None

    def bindParams(self, program: hp.Program, i: int) -> None:
        if self.tensor is None:
            raise RuntimeError("Hit response has not been prepared")
        super().bindParams(program, i)
        program.bindParams(HitQueueOut=self._tensor)

    def run(self, i: int) -> list[hp.Command]:
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
    normalization: float, default=1.0
        Normalization passed down to the hit response.
    blockSize: int, default=128
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
        normalization: float = 1.0,
        polarized: bool = False,
        blockSize: int = 128,
        code: bytes | None = None,
    ) -> None:
        super().__init__()
        # save params
        self._blockSize = blockSize
        self._capacity = capacity
        self._polarized = polarized
        self._response = response

        # prepare response
        c = TraceConfig(capacity, blockSize, capacity, normalization, polarized)
        response.prepare(c)

        # create code if needed
        if code is None:
            preamble = createPreamble(
                BLOCK_SIZE=blockSize,
                HIT_QUEUE_SIZE=capacity,
                HIT_QUEUE_POLARIZED=polarized,
                POLARIZATION=polarized,
            )
            headers = {"response.glsl": response.sourceCode}
            code = compileShader("response.replay.glsl", preamble, headers)
        self._code = code
        self._program = hp.Program(self._code)
        # calculate number of workgroups
        self._groups = -(capacity // -blockSize)

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
    def blockSize(self) -> int:
        """Number of hits processed per work group"""
        return self._blockSize

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

    def run(self, i: int) -> list[hp.Command]:
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


class ValueResponse(SourceCodeMixin):
    """
    Base class for value response functions transforming a hit produced by a
    tracer into a simple numeric value to be consumed by an appropriate
    response functions.

    The corresponding shader code has the following signature:
    ```
    float responseValue(HitItem item)
    ```
    """

    name = "Value Response"

    def __init__(
        self, params: dict[str, type[Structure]] = {}, extra: set[str] = set()
    ) -> None:
        super().__init__(params, extra)

    def prepare(self, config: TraceConfig) -> None:
        """
        Called by e.g. a `Tracer` when binding the response into its program
        notifying it about its configuration to allow it to make adequate
        adjustments to itself beforehand.
        """
        pass


class UniformValueResponse(ValueResponse):
    """
    Response function simulating a perfect isotropic and uniform response.
    """

    def __init__(self) -> None:
        super().__init__()

    # property via descriptor
    sourceCode = ShaderLoader("response.uniform.glsl")


class CustomValueResponse(ValueResponse):
    """
    Response function based on user provided code. Code must define the response
    value function:

    ```
    float responseValue(HitItem item)
    ```

    Parameters
    ----------
    code: str
        User provided source code (GLSL).
    """

    def __init__(self, code: str) -> None:
        super().__init__()
        # add include guards before
        _code = "#ifndef _INCLUDE_RESPONSE_CUSTOM\n"
        _code += "#define _INCLUDE_RESPONSE_CUSTOM\n"
        _code += code
        _code += "#endif\n"
        self._code = _code

    @property
    def sourceCode(self) -> str:
        return self._code


class StoreValueHitResponse(HitResponse):
    """
    Response function storing the result of the given `ValueResponse` along the
    hit's time stamp as `ValueItem` into a queue for further processing.

    Parameters
    ----------
    response: ValueResponse
        Value response function processing hits.
    queue: QueueTensor | None
        Queue in which the `ValueItem` will be stored.
        If `None` creates one during preparations.
    """

    name = "Store Value Response"

    _sourceCode = ShaderLoader("response.value.store.glsl")

    def __init__(
        self, response: ValueResponse, queue: QueueTensor | None = None
    ) -> None:
        super().__init__()
        self._response = response
        self._queue = queue

    @property
    def queue(self) -> QueueTensor | None:
        """Tensor to be filled with `ValueItem`"""
        return self._queue

    @property
    def response(self) -> ValueResponse:
        """Value response function processing hits."""
        return self._response

    @property
    def sourceCode(self) -> str:
        if self.queue is None:
            raise RuntimeError("Response has not been prepared")
        # assemble source code
        guardStart = "#ifndef _INCLUDE_RESPONSE\n#define _INCLUDE_RESPONSE\n"
        guardEnd = "#endif\n"
        preamble = createPreamble(VALUE_QUEUE_SIZE=self.queue.capacity)
        return "\n".join(
            [guardStart, preamble, self.response.sourceCode, self._sourceCode, guardEnd]
        )

    def prepare(self, config: TraceConfig) -> None:
        self.response.prepare(config)
        if self._queue is None:
            self._queue = createValueQueue(config.maxHits)
        elif self.queue.capacity < config.maxHits:
            raise RuntimeError(
                f"Queue not big enough to store hits: Queue has capacity of"
                f"{self.queue.capacity} but {config.maxHits} needed"
            )

    def bindParams(self, program: hp.Program, i: int) -> None:
        if self.queue is None:
            raise RuntimeError("Hit response has not been prepared")
        super().bindParams(program, i)
        program.bindParams(ValueQueueOut=self.queue)
        self.response.bindParams(program, i)


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
    blockSize: int, default=128
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

        _fields_ = [("blockSize", c_uint32), ("nBins", c_uint32)]

    # lazily compile code (shared among all instances, variation via spec consts)
    byte_code = None

    def __init__(
        self,
        *,
        nBins: int,
        nHist: int,
        histIn: hp.Tensor | None = None,
        histOut: hp.Tensor | None = None,
        retrieve: bool = True,
        normalization: float = 1.0,
        blockSize: int = 128,
    ) -> None:
        super().__init__({"Params": self.Params})
        # save params
        self._blockSize = blockSize
        self._nBins = nBins
        self._retrieve = retrieve
        self.setParams(normalization=normalization, nHist=nHist)

        # create code if needed
        if HistogramReducer.byte_code is None:
            HistogramReducer.byte_code = compileShader("estimator.reduce.glsl")
        # create specialization
        spec = HistogramReducer.Constants(blockSize=blockSize, nBins=nBins)
        # compile
        self._program = hp.Program(HistogramReducer.byte_code, bytes(spec))
        self._groups = -(nBins // -blockSize)

        # allocate memory if needed
        n = nBins * nHist
        self._histIn = hp.FloatTensor(n) if histIn is None else histIn
        self._histOut = hp.FloatTensor(nBins) if histOut is None else histOut
        self._buffer = [hp.FloatBuffer(nBins) if retrieve else None for _ in range(2)]
        # bind memory
        self._program.bindParams(HistogramIn=self._histIn, HistogramOut=self._histOut)

    @property
    def blockSize(self) -> int:
        """Number of threads per workgroup"""
        return self._blockSize

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

    def result(self, i: int) -> NDArray | None:
        """The retrieved i-th histogram. `None` if `retrieve`was set to `False`"""
        return self._buffer[i].numpy() if self.retrieve else None

    def run(self, i: int) -> list[hp.Command]:
        self._bindParams(self._program, i)
        cmds = [self._program.dispatch(self._groups)]
        if self.retrieve:
            cmds.append(hp.retrieveTensor(self.histOut, self._buffer[i]))
        return cmds


class HistogramHitResponse(HitResponse):
    """
    Response function producing a histogram using the values produced by the
    given `ValueResponse`. Produces a partial histogram for each workgroup in
    the response function, which need to be reduced into the final histogram in
    a final stage following the tracing.

    Parameters
    ----------
    response: ValueResponse
        Value response function processing hits.
    nBins: int, default=100
        Number of bins in the histogram.
    t0: float, default=0ns
        First bin edge, i.e. the minimum time value a sample has to have to get
        included
    binSize: float, default=1.0ns
        Size of a single bin in unit of time
    normalization: float | None, default=None
        Common factor each bin gets multiplied with. If `None`, uses the value
        from the corresponding tracer during preparation.
    retrieve: bool, default=True
        Wether to retrieve the final histogram, i.e. copying it back to the CPU.
    reduceBlockSize: int, default=128
        Workgroup size of the reduction stage.
    updateResponse: bool, default=True
        If True, when this stage is requested to update by e.g. a pipeline it
        will also cause the response to update.

    Stage Parameters
    ----------------
    t0: float
        First bin edge, i.e. the minimum time value a sample has to have to get
        included
    binSize: float, default=1.0
        Size of a single bin in unit of time
    normalization
        Common factor each bin gets multiplied with

    Note
    ----
    During the update step in a pipeline, the response gets also updated making
    it unnecessary to include as a separate step in most cases. This stage,
    however, ignores any commands produced by run() as it is not clear where to
    put them chronologically.
    """

    name = "Histogram Hit Response"

    # lazily load source code
    _sourceCode = ShaderLoader("response.histogram.glsl")

    class Params(Structure):
        _fields_ = [("t0", c_float), ("binSize", c_float)]

    def __init__(
        self,
        response: ValueResponse,
        *,
        nBins: int = 100,
        t0: float = 0.0 * u.ns,
        binSize: float = 1.0 * u.ns,
        normalization: float | None = None,
        retrieve: bool = True,
        reduceBlockSize: int = 128,
        updateResponse: bool = True,
    ) -> None:
        super().__init__({"ResponseParams": self.Params}, {"normalization"})
        # save params
        self._response = response
        self.setParams(t0=t0, binSize=binSize)
        self._nBins = nBins
        self._normalization = normalization
        self._retrieve = retrieve
        self._reduceBlockSize = reduceBlockSize
        self._updateResponse = updateResponse
        self._reducer = None

    @property
    def nBins(self) -> int:
        """Number of bins in the histogram"""
        return self._nBins

    @property
    def normalization(self) -> float | None:
        """Common factor each bin gets multiplied with"""
        return self._normalization

    @normalization.setter
    def normalization(self, value: float) -> None:
        self._normalization = value
        if self._reducer is not None:
            self._reducer.normalization = value

    @property
    def reduceBlockSize(self) -> int:
        """Workgroup size of the reduction stage."""
        return self._reduceBlockSize

    @property
    def reduceCode(self) -> bytes:
        """
        Compiled source code of the reduction stage. Only available after
        response has been prepared.
        """
        if self._reducer is None:
            raise RuntimeError("Response has not been prepared")
        return self._reducer.code

    @property
    def response(self) -> ValueResponse:
        """Value response function processing hits."""
        return self._response

    @property
    def retrieve(self) -> bool:
        """Wether to retrieve the final histogram"""
        return self._retrieve

    @property
    def sourceCode(self) -> str:
        # we need to define histogram size via macro -> enclose in include guards
        preamble = createPreamble(N_BINS=self.nBins)
        # Include value response
        return "\n".join(
            [
                self.response.sourceCode,
                "#ifndef _INCLUDE_RESPONSE_HISTOGRAM_PREAMBLE",
                "#define _INCLUDE_RESPONSE_HISTOGRAM_PREAMBLE",
                preamble,
                "#endif",
                self._sourceCode,
            ]
        )

    def result(self, i: int) -> NDArray | None:
        """
        The retrieved i-th histogram. `None` if `retrieve` was set to `False` or
        response has not yet been prepared.
        """
        if self._reducer is None:
            return None
        else:
            return self._reducer.result(i)

    def prepare(self, config: TraceConfig) -> None:
        self.response.prepare(config)
        if self.normalization is None:
            self.normalization = config.normalization
        self._reducer = HistogramReducer(
            nBins=self.nBins,
            nHist=-(config.batchSize // -config.blockSize),
            retrieve=self._retrieve,
            normalization=self.normalization,
            blockSize=self._reduceBlockSize,
        )

    def _bindParams(self, program: hp.Program, i: int) -> None:
        super()._bindParams(program, i)
        # connect response and reducer
        program.bindParams(HistogramOut=self._reducer.histIn)

    def update(self, i: int) -> None:
        if self._reducer is None:
            raise RuntimeError("Response has not been prepared")
        super().update(i)
        self._reducer.update(i)
        if self._updateResponse:
            self.response.update(i)

    def run(self, i: int) -> list[hp.Command]:
        return self._reducer.run(i)


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
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
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
    blockSize: int, default=128
        Number of items processed per workgroup during item processing
    reduceBlockSize: int, default=128
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
        blockSize: int = 128,
        reduceBlockSize: int = 128,
        code: bytes | None = None,
    ) -> None:
        super().__init__(
            queue, clearQueue, {"Parameters": self.Params}, {"normalization"}
        )
        # save params
        self._blockSize = blockSize
        self.setParams(t0=t0, binSize=binSize)

        # create reducer
        self._groups = -(queue.capacity // -blockSize)
        self._reducer = HistogramReducer(
            nBins=nBins,
            nHist=self._groups,
            normalization=normalization,
            retrieve=retrieve,
            blockSize=reduceBlockSize,
        )

        # compile code if needed
        if code is None:
            preamble = createPreamble(
                BLOCK_SIZE=blockSize,
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
    def blockSize(self) -> int:
        """Number of threads per workgroup during item processing"""
        return self._blockSize

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
    def reduceBlockSize(self) -> int:
        """Number of threads per workgroup during reduction"""
        return self._reducer.blockSize

    @property
    def resultTensor(self) -> hp.Tensor:
        """Tensor holding the final histogram"""
        return self._reducer.histOut

    @property
    def retrieve(self) -> bool:
        """Wether to retrieve the final histogram"""
        return self._reducer.retrieve

    def result(self, i: int) -> NDArray | None:
        """The retrieved i-th histogram. `None` if `retrieve`was set to `False`"""
        return self._reducer.result(i)

    # Pipeline API

    def update(self, i: int) -> None:
        # also update reducer
        self._reducer.update(i)
        super().update(i)

    def run(self, i: int) -> list[hp.Command]:
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

    def run(self, i: int) -> list[hp.Command]:
        return [
            hp.retrieveTensor(self.queue, self.buffer(i)),
            *([clearQueue(self.queue)] if self.clearQueue else []),
        ]
