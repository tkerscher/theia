from __future__ import annotations

from dataclasses import dataclass
from warnings import warn

import numpy as np
import hephaistos as hp
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import IOQueue

from ctypes import Structure, c_double, c_float, c_int32, c_uint32
from hephaistos.glsl import buffer_reference

from theia.compiler import createPreamble, compileShader, loadShader
from theia.ray import RayModel
from theia.random import RNG
from theia.util import createCType
import theia.units as u

from numpy.typing import ArrayLike


__all__ = [
    "BinReducer",
    "CustomValueResponse",
    "EmptyResponse",
    "HistogramHitResponse",
    "HitRecorder",
    "HitReplay",
    "HitResponse",
    "IntegratingHitResponse",
    "KernelHistogramHitResponse",
    "StoreTimeHitResponse",
    "StoreValueHitResponse",
    "TraceConfig",
    "UniformValueResponse",
    "ValueHitResponse",
    "ValueResponse",
]


def __dir__():
    return __all__


@dataclass
class TraceConfig:
    """
    Description of the tracing.

    Parameters
    ----------
    batchSize: int
        Total number of threads per run.
    capacity: int
        Maximum batch size
    maxHitsPerRay: int
        Maximum number of hits produced per ray.
    normalization: float
        Normalization factor that must be applied to each estimate to get a
        correct estimate.
    """

    batchSize: int
    capacity: int
    maxHitsPerRay: int = 1
    normalization: float = 1.0


class HitResponse(SourceCodeMixin):
    """
    Base class for response functions called for each hit produced by a tracer.
    A response function is a GLSL function consuming a single `HitItem`:

    ```
    void response(HitItem item, uint idx, inout uint dim)
    ```

    `idx` and `dim` are the state of the RNG and can be used to draw random
    numbers.
    """

    name = "Hit Response"

    def __init__(
        self,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
        *,
        nRNGSamples: int = 0,
    ) -> None:
        super().__init__(params, extra)
        self._nRNGSamples = nRNGSamples

    @property
    def nRNGSamples(self) -> int:
        """Number of random numbers drawn per generated hit"""
        return self._nRNGSamples

    def prepare(self, config: TraceConfig, ray: RayModel) -> None:
        """
        Called by e.g. a `Tracer` during initialization to notify this response
        about the tracer's configuration allowing it to make adequate
        adjustments beforehand. If the response is shared among multiple
        tracers, prepare will be called exactly once by each of them. Responses
        should therefore lazily allocate resources at the latest in `bindParams`
        and use `prepare` to keep track of the worst case.
        """
        pass

    def updateConfig(self, config: TraceConfig) -> None:
        """
        Called by e.g. a `Tracer` to notify the response about changes in its
        configuration.
        """
        pass


class EmptyResponse(HitResponse):
    """Empty hit response ignoring all hits"""

    def __init__(self) -> None:
        super().__init__()

    @property
    def sourceCode(self) -> str:
        return loadShader("response/empty.glsl")


class HitRecorder(HitResponse):
    """
    Records hits into a queue, which can be retrieved later on.

    Parameters
    ----------
    capacity: int, default=0
        Maximum number of items the queue can hold. For non-positive values, the
        capacity will be adjusted to the tracer's capacity.
    """

    class ResponseParams(Structure):
        _fields_ = [("_queueAdr", buffer_reference), ("_queueSize", c_uint32)]

    def __init__(self, *, capacity: int = 0) -> None:
        super().__init__({"ResponseParams": HitRecorder.ResponseParams})
        # save params
        self._initCap = capacity
        self.setParam("_queueSize", capacity)
        self._queue: IOQueue | None = None

    def prepare(self, config: TraceConfig, ray: RayModel) -> None:
        maxHits = config.maxHitsPerRay * config.capacity
        if self.capacity > 0 and maxHits > self.capacity:
            warn(
                f"Tracer reported a maximum of {maxHits} hits, but response "
                f"was initialized with a capacity of only {self.capacity}. "
                "This might result in data loss."
            )
        else:
            self.setParam("_queueSize", maxHits)
        # pass 0 as queue capacity to delay init if needed
        initQueueSize = None if self._initCap <= 0 else self.capacity
        self._queue = IOQueue(ray.hitItem, initQueueSize, mode="retrieve")
        if self._queue.initialized:
            assert self._queue.tensor is not None  # to make linter happy...
            self.setParam("_queueAdr", self._queue.tensor.address)

    def updateConfig(self, config: TraceConfig) -> None:
        maxHits = config.maxHitsPerRay * config.capacity
        if self._initCap <= 0 and self.queue is None:
            self.setParam("_queueSize", maxHits)
        elif maxHits > self.capacity:
            warn(
                f"Tracer reported a maximum of {maxHits} hits, but response "
                f"was initialized with a capacity of only {self.capacity}. "
                "This might result in data loss."
            )

    @property
    def capacity(self) -> int:
        """Maximum number of hits that can be saved per run"""
        return self.getParam("_queueSize")

    @property
    def queue(self) -> IOQueue | None:
        """Queue holding the sampled hits"""
        return self._queue

    @property
    def sourceCode(self) -> str:
        return loadShader("response/record.glsl")

    def _finishParams(self, i: int) -> None:
        # lazy creation of queue
        if self.queue is None:
            raise RuntimeError("Response has not been prepared!")
        if not self.queue.initialized:
            self.queue.initialize(self.capacity)
            assert self.queue.tensor is not None  # to make linter happy
            self.setParam("_queueAdr", self.queue.tensor.address)

    def run(self, i: int) -> list[hp.Command]:
        if self.queue is None:
            raise RuntimeError("Response has not been prepared!")
        return self.queue.run(i)


class HitReplay(PipelineStage):
    """
    Reads hits from a queue and feeds each to the given response function.

    Parameters
    ----------
    capacity: int
        Capacity of the underlying queue
    ray: RayModel
        Model describing the structure of hits
    response: HitResponse
        Response the hits gets feed to
    rng: RNG | None, default=None
        Optional random number generator
    """

    class ReplayParams(Structure):
        _fields_ = [("_queueAdr", buffer_reference), ("_queueSize", c_uint32)]

    def __init__(
        self,
        capacity: int,
        ray: RayModel,
        response: HitResponse,
        rng: RNG | None = None,
    ) -> None:
        super().__init__({"ReplayParams": HitReplay.ReplayParams})
        # check if we have a RNG if needed
        if response.nRNGSamples > 0 and rng is None:
            raise ValueError("response requires a RNG but none was given!")
        # prepare response
        self._config = TraceConfig(capacity, capacity, 1)
        response.prepare(self._config, ray)
        # create queue
        self._queue = IOQueue(ray.hitItem, capacity, mode="update")
        assert self._queue.tensor is not None
        self.setParams(_queueAdr=self._queue.tensor.address, _queueSize=capacity)
        # save params
        self._capacity = capacity
        self._ray = ray
        self._response = response
        self._rng = rng

        # create program
        headers = {
            "ray.glsl": ray.sourceCode,
            "response.glsl": response.sourceCode,
            "rng.glsl": "" if rng is None else rng.sourceCode,
        }
        code = compileShader("response/replay.glsl", "", headers)
        self._program = hp.Program(code)

    @property
    def capacity(self) -> int:
        """Maximum number of hits that can be processed per run"""
        return self._capacity

    @property
    def queue(self) -> IOQueue:
        """Queue containing the samples for the next batch"""
        return self._queue

    @property
    def ray(self) -> RayModel:
        """Ray model that defines the hit item used"""
        return self._ray

    @property
    def rng(self) -> RNG | None:
        """The optional RNG used by the response, or `None` if not specified"""
        return self._rng

    @property
    def response(self) -> HitResponse:
        """Response function called for each hit"""
        return self._response

    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all stages involved with this stage in the correct
        order suitable for creating a pipeline-
        """
        if self.rng is not None:
            return [self.rng, self, self.response]
        else:
            return [self, self.response]

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        # if response is shared, we might need to reset the config
        # NOTE: This only works because the response comes after the tracer
        self.response.updateConfig(self._config)

    def run(self, i: int) -> list[hp.Command]:
        for stage in self.collectStages():
            stage.bindParams(self._program, i)
        groups = -(self.capacity // -512)
        return [
            *self.queue.run(i),
            self._program.dispatch(groups),
        ]


class ValueResponse(SourceCodeMixin):
    """
    Base class for value response functions transforming a hit produced by a
    tracer into a simple numeric value to be consumed by an appropriate
    response functions.

    The corresponding shader code has the following signature:

    ```
    float responseValue(HitItem item, uint idx, inout uint dim)
    ```

    `idx` and `dim` are the current RNG state and can be used to draw random
    numbers.
    """

    name = "Value Response"

    def __init__(
        self,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
        *,
        nRNGSamples: int = 0,
    ) -> None:
        super().__init__(params, extra)
        self._nRNGSamples = nRNGSamples

    @property
    def nRNGSamples(self) -> int:
        """Number of random numbers drawn per hit"""
        return self._nRNGSamples

    def prepare(self, config: TraceConfig, ray: RayModel) -> None:
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

    @property
    def sourceCode(self) -> str:
        return loadShader("response/value/uniform.glsl")


class CustomValueResponse(ValueResponse):
    """
    Response function based on user provided code. Code must define the response
    value function:

    ```
    float responseValue(HitItem item, uint idx, inout uint dim)
    ```

    `idx` and `dim` are the current RNG state and can be used to draw random
    numbers.

    Parameters
    ----------
    code: str
        User provided source code (GLSL).
    nRNGSamples: int, default=0
        Number of random numbers drawn per hit.
    """

    def __init__(self, code: str, *, nRNGSamples: int = 0) -> None:
        super().__init__(nRNGSamples=nRNGSamples)
        # add include guards before
        _code = "#ifndef _INCLUDE_RESPONSE_CUSTOM\n"
        _code += "#define _INCLUDE_RESPONSE_CUSTOM\n"
        _code += code
        _code += "#endif\n"
        self._code = _code

    @property
    def sourceCode(self) -> str:
        return self._code


class ValueHitResponse(HitResponse):
    """Base class for hit responses utilizing a `ValueResponse` to process hits"""

    def __init__(
        self,
        response: ValueResponse,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
        *,
        updateResponse: bool,
        nRNGSamples: int = 0,
    ) -> None:
        super().__init__(params, extra, nRNGSamples=response.nRNGSamples + nRNGSamples)
        self._response = response
        self._updateResponse = updateResponse

    @property
    def valueResponse(self) -> ValueResponse:
        """Underlying value response processing hits"""
        return self._response

    def bindParams(self, program: hp.Program | hp.RayTracingPipeline, i: int) -> None:
        super().bindParams(program, i)
        self.valueResponse.bindParams(program, i)

    def prepare(self, config: TraceConfig, ray: RayModel) -> None:
        super().prepare(config, ray)
        self.valueResponse.prepare(config, ray)

    def update(self, i: int) -> None:
        super().update(i)
        if self._updateResponse:
            self.valueResponse.update(i)


class StoreValueHitResponse(ValueHitResponse):
    """
    Response function storing the result of the given `ValueResponse` alongside
    the hit's time and optionally object id into a queue.

    Parameters
    ----------
    capacity: int
        Maximum number of hits that can be stored in the underlying queue
    response: ValueResponse
        Response processing hits
    updateResponse: bool, default=True
        If True and this stage is requested to update by e.g. a pipeline it
        will also cause the response to update.
    storeObjectId: bool, default=True
        Whether to also store the `objectId` of the corresponding hit. Usefull
        if multiple detectors are active at once during tracing.

    Note
    ----
    During the update step in a pipeline, the response gets also updated making
    it unnecessary to include as a separate step in most cases. This stage,
    however, ignores any commands produced by run() as it is not clear where to
    put them chronologically.
    """

    name = "Store Value Response"

    class ResponseParams(Structure):
        _fields_ = [("_queueAdr", buffer_reference), ("_queueSize", c_uint32)]

    def __init__(
        self,
        capacity: int,
        response: ValueResponse,
        *,
        updateResponse: bool = True,
        storeObjectId: bool = True,
    ) -> None:
        super().__init__(
            response,
            {"ResponseParams": StoreValueHitResponse.ResponseParams},
            updateResponse=updateResponse,
        )
        # allocate queue
        fields: list = [("value", c_float), ("time", c_float)]
        if storeObjectId:
            fields.append(("objectId", c_int32))
        item = createCType("Item", fields)
        self._queue = IOQueue(item, capacity, mode="retrieve")
        assert self._queue.tensor is not None
        # save params
        self._capacity = capacity
        self._storeObjectId = storeObjectId
        self.setParams(
            _queueAdr=self._queue.tensor.address,
            _queueSize=capacity,
        )

    @property
    def queue(self) -> IOQueue:
        """Containing the stored hits"""
        return self._queue

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(RESPONSE_STORE_OBJECT_ID=self._storeObjectId)
        code = loadShader("response/value/store.glsl")
        return "\n".join(
            [
                preamble,
                '#line 1 "value.glsl"',
                self.valueResponse.sourceCode,
                '#line 1 "response/value/store.glsl"',
                code,
            ]
        )

    def run(self, i: int) -> list[hp.Command]:
        return self._queue.run(i)


class StoreTimeHitResponse(ValueHitResponse):
    """
    Treats the value returned by the given `ValueResponse` as detection
    probability and randomly decides to accept the corresponding hit based on
    that probability. Only stores the time stamp and optionally the
    corresponding object id.

    Parameters
    ----------
    capacity: int
        Maximum number of hits that can be stored in the underlying queue
    response: ValueResponse
        Response processing hits
    updateResponse: bool, default=True
        If True and this stage is requested to update by e.g. a pipeline it
        will also cause the response to update.
    storeObjectId: bool, default=True
        Whether to also store the `objectId` of the corresponding hit. Usefull
        if multiple detectors are active at once during tracing.

    Note
    ----
    During the update step in a pipeline, the response gets also updated making
    it unnecessary to include as a separate step in most cases. This stage,
    however, ignores any commands produced by run() as it is not clear where to
    put them chronologically.
    """

    name = "Store Time Hit Response"

    class ResponseParams(Structure):
        _fields_ = [("_queueAdr", buffer_reference), ("_queueSize", c_uint32)]

    def __init__(
        self,
        capacity: int,
        response: ValueResponse,
        *,
        updateResponse: bool = True,
        storeObjectId: bool = True,
    ) -> None:
        super().__init__(
            response,
            {"ResponseParams": StoreValueHitResponse.ResponseParams},
            updateResponse=updateResponse,
            nRNGSamples=1,
        )
        # allocate queue
        if storeObjectId:
            fields = [("time", c_float), ("objectId", c_int32)]
        else:
            fields = [("time", c_float)]
        item = createCType("Item", fields)
        self._queue = IOQueue(item, capacity, mode="retrieve")
        assert self._queue.tensor is not None
        # save params
        self._capacity = capacity
        self._storeObjectId = storeObjectId
        self.setParams(
            _queueAdr=self._queue.tensor.address,
            _queueSize=capacity,
        )

    @property
    def queue(self) -> IOQueue:
        """Containing the stored hits"""
        return self._queue

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(RESPONSE_STORE_OBJECT_ID=self._storeObjectId)
        code = loadShader("response/value/timestamp.glsl")
        return "\n".join(
            [
                preamble,
                '#line 1 "value.glsl"',
                self.valueResponse.sourceCode,
                '#line 1 "response/value/timestamp.glsl"',
                code,
            ]
        )

    def run(self, i: int) -> list[hp.Command]:
        return self._queue.run(i)


class BinReducer(PipelineStage):
    """
    Utility stage for reducing a 2D array along the first axis. This can be used
    for instance for combining a set of histograms into a single one. For small
    arrays this happens on the CPU, whereas for large arrays this happens on the
    GPU.
    """

    name = "Bin Reducer"

    def __init__(
        self,
        *,
        binCount: int,
        bufferCount: int,
        normalization: float = 1.0,
        useDoublePrecision: bool = True,
    ) -> None:
        # save params
        self._normalization = normalization
        self._binCount = binCount
        self._bufferCount = bufferCount
        # switch on double precision
        if useDoublePrecision:
            Buffer, Tensor = hp.DoubleBuffer, hp.DoubleTensor
        else:
            Buffer, Tensor = hp.FloatBuffer, hp.FloatTensor
        self._bufferIn = Tensor(binCount * bufferCount)
        hp.execute(hp.clearTensor(self._bufferIn))
        # reduce small amounts on CPU
        self._reduceGPU = binCount >= 32 and bufferCount >= 32
        if self._reduceGPU:
            # create I/O
            params = [
                ("_bufferInAdr", buffer_reference),
                ("_bufferOutAdr", buffer_reference),
                ("_binCount", c_uint32),
                ("_bufferCount", c_uint32),
                ("_norm", c_double if useDoublePrecision else c_float),
            ]
            super().__init__({"Params": createCType("Params", params)})
            # allocate memory
            self._bufferOut = Tensor(binCount)
            self._buffers = [Buffer(binCount) for _ in range(2)]
            self.setParams(
                _bufferInAdr=self._bufferIn.address,
                _bufferOutAdr=self._bufferOut.address,
                _binCount=binCount,
                _bufferCount=bufferCount,
                _norm=normalization,
            )
            # create program
            preamble = createPreamble(USE_DOUBLE=useDoublePrecision)
            code = compileShader("response/reduce.glsl", preamble)
            self._program = hp.Program(code)
        else:
            super().__init__({}, {"normalization"})
            self._buffers = [Buffer(binCount * bufferCount) for _ in range(2)]

    @property
    def binCount(self) -> int:
        """Number of bins"""
        return self._binCount

    @property
    def bufferCount(self) -> int:
        """Number of buffers in input tensor"""
        return self._bufferCount

    @property
    def bufferIn(self) -> hp.Tensor:
        """Tensor containing input buffer"""
        return self._bufferIn

    @property
    def normalization(self) -> float:
        """Common normalization factor all bins are multiplied with"""
        return self._normalization

    @normalization.setter
    def normalization(self, value: float) -> None:
        self._normalization = value
        if self._reduceGPU:
            self.setParam("_norm", value)

    def result(self, i: int) -> ArrayLike:
        """Returns the final reduced buffer"""
        if self._reduceGPU:
            return self._buffers[i].numpy()
        else:
            # final reduction on CPU
            buffer = np.asarray(self._buffers[i].numpy(), np.float64)
            buffer = buffer.reshape((self.bufferCount, self.binCount))
            return buffer.sum(0) * self.normalization

    def run(self, i: int) -> list[hp.Command]:
        if self._reduceGPU:
            self.bindParams(self._program, i)
            groups = -(self.binCount // -128)
            return [
                hp.flushMemory(),
                self._program.dispatch(groups),
                hp.retrieveTensor(self._bufferOut, self._buffers[i]),
                hp.clearTensor(self._bufferIn),
            ]
        else:
            return [
                hp.retrieveTensor(self._bufferIn, self._buffers[i]),
                hp.clearTensor(self._bufferIn),
            ]


class IntegratingHitResponse(ValueHitResponse):
    """
    Response function summing up all created hits.

    Parameters
    ----------
    response: ValueResponse
        Value response processing hits.
    detectorCount: int | None, default=None
        Number of distinct detectors as identified by their `objectId`. None of
        them must be larger than `objectId` minus one, but you may leave gaps.
        If `None`, all hits will be integrated into one value regardless of
        their `objectId`.
    bufferCount: int, default=128
        Number of internal buffers used to increase performance. Tuning this
        value can further increase performance.
    useDoublePrecision: bool, default=True
        Whether to use double precision while summing up individual hits.
        Using double precision results in less numerical error, but may impact
        performance.
    updateResponse: bool, default=True
        If True, when this stage is requested to update by e.g. a pipeline it
        will also cause the response to update.

    Note
    ----
    During the update step in a pipeline, if `updateResponse` is True the
    response gets also updated making it unnecessary to include as a separate
    step in most cases. This stage, however, ignores any commands produced by
    response.run() as it is not clear where to put them chronologically.
    """

    name = "Integrating Hit Response"

    class ResponseParams(Structure):
        _fields_ = [
            ("_bufferAdr", buffer_reference),
            ("_bufferCount", c_uint32),
            ("_binCount", c_uint32),
        ]

    def __init__(
        self,
        response: ValueResponse,
        *,
        detectorCount: int | None = None,
        bufferCount: int = 128,
        useDoublePrecision: bool = True,
        updateResponse: bool = True,
    ) -> None:
        super().__init__(
            response,
            {"ResponseParams": IntegratingHitResponse.ResponseParams},
            updateResponse=updateResponse,
        )
        self._detCount = detectorCount
        self._useDouble = useDoublePrecision
        detectorCount = detectorCount or 1
        # allocate buffers via reducer
        self._reducer = BinReducer(
            binCount=detectorCount,
            bufferCount=bufferCount,
            useDoublePrecision=useDoublePrecision,
        )
        # set params
        self.setParams(
            _bufferAdr=self._reducer.bufferIn.address,
            _bufferCount=bufferCount,
            _binCount=detectorCount,
        )

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            RESPONSE_INTEGRATE_ALL=self._detCount is None,
            BinBuffer="DoubleBuffer" if self._useDouble else "FloatBuffer",
        )
        code = loadShader("response/value/integrate.glsl")
        return "\n".join(
            [
                preamble,
                '#line 1 "value.glsl"',
                self.valueResponse.sourceCode,
                '#line 1 "response/value/integrate.glsl"',
                code,
            ]
        )

    def result(self, i: int) -> ArrayLike:
        """The result of the i-th buffer"""
        return self._reducer.result(i)

    def update(self, i: int) -> None:
        super().update(i)
        self._reducer.update(i)

    def run(self, i: int) -> list[hp.Command]:
        return self._reducer.run(i)


class HistogramHitResponse(ValueHitResponse):
    """
    Response function creating a histogram from the values produced by the
    given `ValueResponse` for each hit binned by their respective hit time.
    Hits outside the range of the histogram are ignored.

    Parameters
    ----------
    response: ValueResponse
        Value response processing hits.
    binCount: int, default=100
        Number of bins in the histogram
    t0: float, default=0.0
        Left-most edge of the histogram.
    binSize: float, default=1.0ns
        Width of each bin.
    normalization: float | None, default=None
        Common normalization factor each bin gets multiplied with. If `None`,
        uses the normalization constant reported by the tracer.
    bufferCount: int, default=128
        Number of internal buffers used to increase performance. Tuning this
        value can further increase performance.
    useDoublePrecision: bool, default=True
        Whether to use double precision while summing up partial histograms.
        Using double precision results in less numerical error, but may impact
        performance.
    updateResponse: bool, default=True
        If True, when this stage is requested to update by e.g. a pipeline it
        will also cause the response to update.

    Note
    ----
    During the update step in a pipeline, if `updateResponse` is True the
    response gets also updated making it unnecessary to include as a separate
    step in most cases. This stage, however, ignores any commands produced by
    response.run() as it is not clear where to put them chronologically.
    """

    name = "Histogram Response"

    class ResponseParams(Structure):
        _fields_ = [
            ("_bufferAdr", buffer_reference),
            ("_binCount", c_uint32),
            ("_histCount", c_uint32),
            ("t0", c_float),
            ("binSize", c_float),
        ]

    def __init__(
        self,
        response: ValueResponse,
        *,
        binCount: int = 100,
        t0: float = 0.0,
        binSize: float = 1.0 * u.ns,
        normalization: float | None = None,
        bufferCount: int = 128,
        useDoublePrecision: bool = True,
        updateResponse: bool = True,
    ) -> None:
        super().__init__(
            response,
            {"ResponseParams": HistogramHitResponse.ResponseParams},
            {"normalization"},
            updateResponse=updateResponse,
        )
        # let reducer allocate memory
        self._reducer = BinReducer(
            binCount=binCount,
            bufferCount=bufferCount,
            normalization=normalization or 1.0,
            useDoublePrecision=useDoublePrecision,
        )
        # save params
        self._normalization = normalization
        self._useDouble = useDoublePrecision
        self.setParams(
            _bufferAdr=self._reducer.bufferIn.address,
            _binCount=binCount,
            _histCount=bufferCount,
            t0=t0,
            binSize=binSize,
        )

    @property
    def binCount(self) -> int:
        """Number of bins in the histogram"""
        return self._reducer.binCount

    @property
    def normalization(self) -> float:
        """Normalization factor applied to all bins"""
        return self._reducer.normalization

    @normalization.setter
    def normalization(self, value: float) -> None:
        self._normalization = value
        self._reducer.normalization = value

    @normalization.deleter
    def normalization(self) -> None:
        self._normalization = None
        self._reducer.normalization = 1.0

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            BinBuffer="DoubleBuffer" if self._useDouble else "FloatBuffer",
        )
        code = loadShader("response/histogram/simple.glsl")
        return "\n".join(
            [
                preamble,
                '#line 1 "value.glsl"',
                self.valueResponse.sourceCode,
                '#line 1 "response/histogram/simple.glsl"',
                code,
            ]
        )

    def result(self, i: int) -> ArrayLike:
        """The result of the i-th buffer"""
        return self._reducer.result(i)

    def prepare(self, config: TraceConfig, ray: RayModel) -> None:
        super().prepare(config, ray)
        if self._normalization is None:
            self._reducer.normalization = config.normalization

    def updateConfig(self, config: TraceConfig) -> None:
        super().updateConfig(config)
        if self._normalization is None:
            self._reducer.normalization = config.normalization

    def update(self, i: int) -> None:
        super().update(i)
        self._reducer.update(i)

    def run(self, i: int) -> list[hp.Command]:
        return self._reducer.run(i)


class KernelHistogramHitResponse(ValueHitResponse):
    """
    Response function creating a histogram from the values produced by the
    given `ValueResponse`. The contribution from each bin get smeared out using
    a gaussian kernel of given shape.
    Hits outside the range of the histogram are ignored.

    Parameters
    ----------
    response: ValueResponse
        Value response processing hits.
    binCount: int, default=100
        Number of bins in the histogram
    t0: float, default=0.0
        Left-most edge of the histogram.
    binSize: float, default=1.0ns
        Width of each bin.
    kernelBandwidth: float
        Bandwidth of the kernel. Corresponds to the standard deviation for
        gaussian kernel.
    kernelSupport: float
        Limits the range the kernel affects to +/- the support.
    normalization: float | None, default=None
        Common normalization factor each bin gets multiplied with. If `None`,
        uses the normalization constant reported by the tracer.
    bufferCount: int, default=128
        Number of internal buffers used to increase performance. Tuning this
        value can further increase performance.
    useDoublePrecision: bool, default=True
        Whether to use double precision while summing up partial histograms.
        Using double precision results in less numerical error, but may impact
        performance.
    updateResponse: bool, default=True
        If True, when this stage is requested to update by e.g. a pipeline it
        will also cause the response to update.

    Note
    ----
    During the update step in a pipeline, the response gets also updated making
    it unnecessary to include as a separate step in most cases. This stage,
    however, ignores any commands produced by run() as it is not clear where to
    put them chronologically.

    See Also
    --------
    theia.response.HistogramHitResponse
    """

    name = "Kernel Histogram Response"

    class ResponseParams(Structure):
        _fields_ = [
            ("_bufferAdr", buffer_reference),
            ("_binCount", c_uint32),
            ("_histCount", c_uint32),
            ("t0", c_float),
            ("binSize", c_float),
            ("kernelBandwidth", c_float),
            ("kernelSupport", c_float),
        ]

    def __init__(
        self,
        response: ValueResponse,
        *,
        binCount: int = 100,
        t0: float = 0.0,
        binSize: float = 1.0 * u.ns,
        kernelBandwidth: float = 1.0 * u.ns,
        kernelSupport: float = 3.0 * u.ns,
        normalization: float | None = None,
        bufferCount: int = 128,
        useDoublePrecision: bool = True,
        updateResponse: bool = True,
    ) -> None:
        super().__init__(
            response,
            {"ResponseParams": KernelHistogramHitResponse.ResponseParams},
            {"normalization"},
            updateResponse=updateResponse,
        )
        # let reducer allocate memory
        self._reducer = BinReducer(
            binCount=binCount,
            bufferCount=bufferCount,
            normalization=normalization or 1.0,
            useDoublePrecision=useDoublePrecision,
        )
        # save params
        self._useDouble = useDoublePrecision
        self._normalization = normalization
        self.setParams(
            _bufferAdr=self._reducer.bufferIn.address,
            _binCount=binCount,
            _histCount=bufferCount,
            t0=t0,
            binSize=binSize,
            kernelBandwidth=kernelBandwidth,
            kernelSupport=kernelSupport,
        )

    @property
    def binCount(self) -> int:
        """Number of bins in the histogram"""
        return self._reducer.binCount

    @property
    def normalization(self) -> float:
        """Normalization factor applied to all bins"""
        return self._reducer.normalization

    @normalization.setter
    def normalization(self, value: float) -> None:
        self._normalization = value
        self._reducer.normalization = value

    @normalization.deleter
    def normalization(self) -> None:
        self._normalization = None
        self._reducer.normalization = 1.0

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            BinBuffer="DoubleBuffer" if self._useDouble else "FloatBuffer",
        )
        code = loadShader("response/histogram/kernel.glsl")
        return "\n".join(
            [
                preamble,
                '#line 1 "value.glsl"',
                self.valueResponse.sourceCode,
                '#line 1 "response/histogram/kernel.glsl"',
                code,
            ]
        )

    def result(self, i: int) -> ArrayLike:
        """The result of the i-th buffer"""
        return self._reducer.result(i)

    def prepare(self, config: TraceConfig, ray: RayModel) -> None:
        super().prepare(config, ray)
        if self._normalization is None:
            self._reducer.normalization = config.normalization

    def updateConfig(self, config: TraceConfig) -> None:
        super().updateConfig(config)
        if self._normalization is None:
            self._reducer.normalization = config.normalization

    def update(self, i: int) -> None:
        super().update(i)
        self._reducer.update(i)

    def run(self, i: int) -> list[hp.Command]:
        return self._reducer.run(i)
