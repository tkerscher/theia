from __future__ import annotations
from dataclasses import dataclass
from typing import List
from warnings import warn

import hephaistos as hp
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import IOQueue, QueueBuffer, QueueTensor, QueueView, clearQueue

from ctypes import Structure, c_float, c_int32, c_uint32

from theia.camera import Camera
from theia.light import WavelengthSource
from theia.random import RNG
from theia.util import ShaderLoader, compileShader, createPreamble
import theia.units as u

from numpy.typing import NDArray


__all__ = [
    "createHitTimeQueue",
    "createValueQueue",
    "CameraHitResponseItem",
    "CameraHitResponseSampler",
    "CustomValueResponse",
    "EmptyResponse",
    "Estimator",
    "HistogramEstimator",
    "HistogramReducer",
    "HitItem",
    "HitRecorder",
    "HitReplay",
    "HitResponse",
    "HitTimeAndIdItem",
    "HitTimeItem",
    "KernelHistogramHitResponse",
    "PolarizedCameraHitResponseItem",
    "PolarizedHitItem",
    "SampleValueResponse",
    "StoreTimeHitResponse",
    "StoreValueHitResponse",
    "UniformValueResponse",
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
        ("objectId", c_int32),
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
        ("objectId", c_int32),
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
    capacity: int
        Maximum batch size
    maxHitsPerThread: int
        Maximum number of hits produced per thread.
    normalization: float
        Normalization factor that must be applied to each sample to get a
        correct estimate.
    polarized: bool
        True, if hits contain polarization information.
    """

    batchSize: int
    blockSize: int
    capacity: int
    maxHitsPerThread: int = 1
    normalization: float = 1.0
    polarized: bool = False


class HitResponse(SourceCodeMixin):
    """
    Base class for response functions called for each hit produced by a tracer.
    A response function is a GLSL function consuming a single `HitItem`:

    ```
    void response(HitItem item, uint idx, inout uint dim)
    ```

    `idx` and `dim` are the state of the RNG and can be used to draw random
    numbers. Additionally, the following two functions must be present:

    ````
    void initResponse()
    void finalizeResponse()
    ```

    These will get called at the start and end of the simulation respectively.
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

    def prepare(self, config: TraceConfig) -> None:
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

    # sourceCode via descriptor
    sourceCode = ShaderLoader("response.empty.glsl")


class HitRecorder(HitResponse):
    """
    Records hits onto a queue, which can be retrieved later on.
    Should be placed in the pipeline at a later stage than the tracing.

    Parameters
    ----------
    capacity: int, default=0
        Maximum number of items to record per batch. If a non-positive number
        is provided, the capacity will be adjusted to the tracer's capacity.
    polarized: bool, default=False
        Whether to save the polarization state.

    Note
    ----
    If the hit source (e.g. a tracer) does not produce polarized samples, the
    polarization reference will be the null vector.
    """

    _sourceCode = ShaderLoader("response.record.glsl")

    def __init__(self, *, capacity: int = 0, polarized: bool = False) -> None:
        super().__init__()
        # save params
        self._capacity = capacity
        self._polarized = polarized
        item = PolarizedHitItem if polarized else HitItem
        self._queue = IOQueue(item, mode="retrieve")
        if capacity > 0:
            self._queue.initialize(capacity)

    def prepare(self, config: TraceConfig) -> None:
        maxHits = config.maxHitsPerThread * config.capacity
        if self.queue.initialized and maxHits > self.capacity:
            warn(
                f"tracer reported maxHits: {maxHits}, "
                f"but response was initialized with capacity: {self.capacity}"
            )
        elif not self.queue.initialized:
            # update worst case
            self._capacity = max(self._capacity, maxHits)

    def updateConfig(self, config: TraceConfig) -> None:
        maxHits = config.batchSize * config.maxHitsPerThread
        if not self.queue.initialized:
            self._capacity = max(maxHits, self._capacity)
        elif maxHits > self.capacity:
            warn(
                f"tracer reported maxHits: {maxHits}, "
                f"but response was initialized with capacity: {self.capacity}"
            )

    @property
    def capacity(self) -> int:
        """Maximum number of hits that can be saved per run"""
        return self._capacity

    @property
    def polarized(self) -> bool:
        """Whether to save the polarization state"""
        return self._polarized

    @property
    def queue(self) -> IOQueue:
        """Queue holding the sampled hits"""
        return self._queue

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            HIT_QUEUE_SIZE=self.capacity,
            HIT_QUEUE_POLARIZED=self.polarized,
        )
        return preamble + self._sourceCode

    def bindParams(self, program: hp.Program, i: int) -> None:
        if not self.queue.initialized:
            if self.capacity <= 0:
                raise RuntimeError(f"Cannot create a queue of size {self.capacity}")
            self.queue.initialize(self.capacity)
        super().bindParams(program, i)
        program.bindParams(HitQueueOut=self.queue.tensor)

    def run(self, i: int) -> list[hp.Command]:
        return self.queue.run(i)


class HitReplay(PipelineStage):
    """
    Reads hits from a queue and feeds each to the given response function.

    Parameters
    ----------
    capacity: int
        Maximum number of hits that can be processed per run
    response: HitResponse
        Response to be called on each hit
    rng: RNG | None, default=None
        The random number generator used by the response. May be `None` if the
        response does not require random numbers.
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
        rng: RNG | None = None,
        normalization: float = 1.0,
        polarized: bool = False,
        blockSize: int = 128,
        code: bytes | None = None,
    ) -> None:
        # check if we have a rng if needed
        if response.nRNGSamples > 0 and rng is None:
            raise ValueError("response requires a rng but none was given!")

        super().__init__()
        # save params
        self._blockSize = blockSize
        self._capacity = capacity
        self._polarized = polarized
        self._response = response
        self._rng = rng

        # prepare response
        self._config = TraceConfig(
            capacity,
            blockSize,
            capacity,
            1,
            normalization,
            polarized,
        )
        response.prepare(self._config)

        # create code if needed
        if code is None:
            preamble = createPreamble(
                BLOCK_SIZE=blockSize,
                HIT_QUEUE_SIZE=capacity,
                HIT_QUEUE_POLARIZED=polarized,
                POLARIZATION=polarized,
            )
            headers = {
                "response.glsl": response.sourceCode,
                "rng.glsl": rng.sourceCode if rng is not None else "",
            }
            code = compileShader("response.replay.glsl", preamble, headers)
        self._code = code
        self._program = hp.Program(self._code)
        # calculate number of workgroups
        self._groups = -(capacity // -blockSize)

        # create queue
        item = PolarizedHitItem if polarized else HitItem
        self._queue = IOQueue(item, capacity, mode="update")
        # bind memory
        self._program.bindParams(HitQueueIn=self.queue.tensor)

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
    def queue(self) -> IOQueue:
        """Queue containing the samples for the next batch"""
        return self._queue

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
        self._bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        if self.rng is not None:
            self.rng.bindParams(self._program, i)
        return [
            *self.queue.run(i),
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

    name = "Store Value Response"

    _sourceCode = ShaderLoader("response.value.store.glsl")

    def __init__(
        self,
        response: ValueResponse,
        queue: QueueTensor | None = None,
        *,
        updateResponse: bool = True,
    ) -> None:
        super().__init__(nRNGSamples=response.nRNGSamples)
        self._response = response
        self._queue = queue
        self._updateResponse = updateResponse

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

    def bindParams(self, program: hp.Program, i: int) -> None:
        if self.queue is None:
            raise RuntimeError("Hit response has not been prepared")
        super().bindParams(program, i)
        program.bindParams(ValueQueueOut=self.queue)
        self.response.bindParams(program, i)

    def prepare(self, config: TraceConfig) -> None:
        self.response.prepare(config)
        maxHits = config.maxHitsPerThread * config.capacity
        if self._queue is None:
            self._queue = createValueQueue(maxHits)
        elif self._queue.capacity < maxHits:
            raise RuntimeError(
                f"Queue not big enough to store hits: Queue has capacity of"
                f"{self._queue.capacity} but {maxHits} needed"
            )

    def updateConfig(self, config: TraceConfig) -> None:
        maxHits = config.maxHitsPerThread * config.capacity
        if self.queue is not None and self.queue.capacity < maxHits:
            raise RuntimeError(
                f"Queue not big enough to store hits: Queue has capacity of"
                f"{self.queue.capacity} but {maxHits} needed"
            )

    def update(self, i):
        super().update(i)
        if self._updateResponse:
            self.response.update(i)


class HitTimeItem(Structure):
    """Item used by `StoreTimeHitResponse` storing only time"""

    _fields_ = [("time", c_float)]


class HitTimeAndIdItem(Structure):
    """Item used by `StoreTimeHitResponse` storing both time and object id"""

    _fields_ = [("time", c_float), ("objectId", c_int32)]


def createHitTimeQueue(capacity: int, storeObjectId: bool = True) -> QueueTensor:
    """
    Creates a new `QueueTensor` of the specified capacity to be used with
    `StoreHitTimeResponse`.

    Parameters
    ----------
    capacity: int
        Maximum number of items in the queue
    storeObjectId: bool, default=True
        Whether to also store the object id of the hit.
    """
    item = HitTimeAndIdItem if storeObjectId else HitTimeItem
    queue = QueueTensor(item, capacity)
    hp.execute(clearQueue(queue))  # sets counter to zero
    return queue


class StoreTimeHitResponse(HitResponse):
    """
    Response function randomly storing the arrival time and optionally the
    object id of the hits based on the probability calculated by the provided
    `ValueResponse`. This is especially usefull if the corresponding tracer is
    configured to trace single photons. In that case the stored times correspond
    to single detected photons and the `ValueResponse` can be used to model a
    detector response.

    Parameters
    ----------
    response: ValueResponse
        Value response processing the hits producing the acceptance probability.
    queue: QueueTensor | None, default=None
        Tensor in which the items will be stored. If `None` a new one will be
        allocated during the tracer's preparation step.
    retrieve: bool, default=True
        Whether to retrieve the results from the device after tracing.
    storeObjectId: bool, default=True
        Whether to also store the `objectId` of the corresponding hit. Usefull
        if multiple detectors are active at once during tracing.
    updateResponse: bool, default=True
        If True and this stage is requested to update by e.g. a pipeline it
        will also cause the response to update.

    Note
    ----
    During the update step in a pipeline, the response gets also updated making
    it unnecessary to include as a separate step in most cases. This stage,
    however, ignores any commands produced by run() as it is not clear where to
    put them chronologically.
    """

    name = "Store Time Hit Response"

    _sourceCode = ShaderLoader("response.time.store.glsl")

    def __init__(
        self,
        response: ValueResponse,
        queue: QueueTensor | None = None,
        *,
        retrieve: bool = True,
        storeObjectId: bool = True,
        updateResponse: bool = True,
    ) -> None:
        super().__init__(nRNGSamples=response.nRNGSamples + 1)
        self._response = response
        self._queue = queue
        self._retrieve = retrieve
        self._storeObjectId = storeObjectId
        self._updateResponse = updateResponse
        self._buffers = None

    @property
    def queue(self) -> QueueTensor | None:
        """Tensor to be filled with the arrival times and optionally the object ids"""
        return self._queue

    @property
    def response(self) -> ValueResponse:
        """Value response function processing hits."""
        return self._response

    @property
    def retrieve(self) -> bool:
        """Whether the stored times are retrieved from the GPU"""
        return self._retrieve

    @property
    def sourceCode(self) -> str:
        if self.queue is None:
            raise RuntimeError("Response has not been prepared!")
        # assemble source code
        guardStart = "#ifndef _INCLUDE_RESPONSE\n#define _INCLUDE_RESPONSE\n"
        guardEnd = "#endif\n"
        preamble = createPreamble(
            VALUE_QUEUE_SIZE=self.queue.capacity,
            RESPONSE_STORE_OBJECT_ID=self._storeObjectId,
        )
        return "\n".join(
            [guardStart, preamble, self.response.sourceCode, self._sourceCode, guardEnd]
        )

    def result(self, i: int) -> QueueView | None:
        """
        The retrieved i-th queue. `None` if `retrieve` is `False` or the
        response has not yet been prepared.
        """
        if not self.retrieve or self._buffers is None:
            return None
        else:
            return self._buffers[i].view

    def prepare(self, config: TraceConfig) -> None:
        self.response.prepare(config)
        maxHits = config.maxHitsPerThread * config.capacity
        if self._queue is None:
            self._queue = createHitTimeQueue(maxHits, self._storeObjectId)
        elif self._queue.capacity < maxHits:
            warn(
                f"The provided queue's capacity ({self._queue.capacity}) is less "
                f"than the max hit count reported by the tracer ({maxHits}). "
                f"Data corruption may occur!"
            )
        expItem = HitTimeAndIdItem if self._storeObjectId else HitTimeItem
        if self._queue.item is not expItem:
            raise RuntimeError(
                f"Unexpected queue item: {self._queue.item}. Expected {expItem}"
            )
        if self.retrieve:
            # allocate double buffered buffers
            self._buffers = [
                QueueBuffer(expItem, self._queue.capacity) for _ in range(2)
            ]

    def updateConfig(self, config: TraceConfig) -> None:
        # It's save to assume maxHits won't change so nothing to do here
        pass

    def bindParams(self, program: hp.Program, i: int) -> None:
        if self.queue is None:
            raise RuntimeError("Hit response has not been prepared")
        super().bindParams(program, i)
        program.bindParams(ValueQueueOut=self.queue)
        self.response.bindParams(program, i)

    def update(self, i: int) -> None:
        super().update(i)
        if self._updateResponse:
            self.response.update(i)

    def run(self, i: int) -> list[hp.Command]:
        if self.retrieve:
            if self._buffers is None:
                raise RuntimeError("Response has not been prepared!")
            return [
                hp.retrieveTensor(self._queue, self._buffers[i]),
                clearQueue(self._queue),  # reset counter to zero
            ]
        else:
            return []


class SampleValueResponse(HitResponse):
    """
    Response function storing the result of the given `ValueResponse` ordered by
    the thread id and retrieving them to the host. Meant for testing value
    response functions.

    Parameters
    ----------
    response: ValueResponse
        Value response function processing hits.
    updateResponse: bool, default=True
        If True, when this stage is requested to update by e.g. a pipeline it
        will also cause the response to update.

    Note
    ----
    For now expects a single hit per thread.

    During the update step in a pipeline, if `updateResponse` is True the
    response gets also updated making it unnecessary to include as a separate
    step in most cases. This stage, however, ignores any commands produced by
    response.run() as it is not clear where to put them chronologically.
    """

    name = "Sample Value Response"

    _sourceCode = ShaderLoader("response.value.sample.glsl")

    def __init__(self, response: ValueResponse, *, updateResponse: bool = True) -> None:
        super().__init__(nRNGSamples=response.nRNGSamples)
        self._response = response
        self._updateResponse = updateResponse
        # will later be created during prepare()
        self._tensor = None
        self._buffer = None

    @property
    def response(self) -> ValueResponse:
        """Value response function processing hits."""
        return self._response

    @property
    def sourceCode(self) -> str:
        if self._tensor is None:
            raise RuntimeError("Response has not been prepared!")
        # assemble source code
        guardStart = "#ifndef _INCLUDE_RESPONSE\n#define _INCLUDE_RESPONSE\n"
        guardEnd = "#endif"
        # preamble = createPreamble(VALUE_QUEUE_SIZE=self._tensor.size)
        return "\n".join(
            [guardStart, self.response.sourceCode, self._sourceCode, guardEnd]
        )

    def bindParams(self, program, i):
        if self._tensor is None:
            raise RuntimeError("Response has not been prepared!")
        super().bindParams(program, i)
        program.bindParams(ValueQueueOut=self._tensor)
        self.response.bindParams(program, i)

    def prepare(self, config):
        if config.maxHitsPerThread != 1:
            raise ValueError("This sampler only supports a single hit per thread!")
        self.response.prepare(config)
        self._tensor = hp.FloatTensor(config.batchSize)
        self._buffer = [hp.FloatBuffer(config.batchSize) for _ in range(2)]

    def result(self, i: int) -> NDArray:
        """Returns the result of the given config"""
        if self._buffer is None:
            raise RuntimeError("Response has not been prepared!")
        return self._buffer[i].numpy()

    def update(self, i):
        super().update(i)
        if self._updateResponse:
            self.response.update(i)

    def run(self, i) -> list[hp.Command]:
        if self._tensor is None:
            raise RuntimeError("Response has not been prepared!")
        return [hp.retrieveTensor(self._tensor, self._buffer[i])]


class CameraHitResponseItem(Structure):
    _fields_ = [
        ("position", c_float * 3),
        ("direction", c_float * 3),
        ("normal", c_float * 3),
        ("wavelength", c_float),
        ("timeDelta", c_float),
        ("contrib", c_float),
    ]


class PolarizedCameraHitResponseItem(Structure):
    _fields_ = [
        ("position", c_float * 3),
        ("direction", c_float * 3),
        ("normal", c_float * 3),
        ("wavelength", c_float),
        ("timeDelta", c_float),
        ("contrib", c_float),
        ("polarizationRef", c_float * 3),
        ("stokes", c_float * 3),
    ]


class CameraHitResponseSampler(PipelineStage):
    """
    Util class calling the given hit response with hits sampled from the given
    camera and storing the results.

    Parameters
    ----------
    batchSize: int
        Number of samples drawn per run
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from
    camera: Camera
        Camera to sample hits from
    response: HitResponse
        Response processing the sampled hits
    rng: RNG | None, default=None
        The random number generator used for sampling. May be `None` if both
        `camera` and `wavelengthSource` do not require random numbers.
    polarized: bool, default=True
        Whether to sample and save polarization information
    blockSize: int, default=128
        Number of samples drawn per work group
    code: bytes | None, default=None
        Compiled source code. If `None`, the byte code get's compiled from
        source. Note, that the compiled code is not checked. You must ensure
        it matches the configuration.
    """

    name = "Camera Hit Response Sampler"

    def __init__(
        self,
        batchSize: int,
        wavelengthSource: WavelengthSource,
        camera: Camera,
        response: HitResponse,
        *,
        rng: RNG | None = None,
        polarized: bool = False,
        blockSize: int = 128,
        code: bytes | None = None,
    ) -> None:
        super().__init__()

        # save params
        self._batchSize = batchSize
        self._camera = camera
        self._response = response
        self._photons = wavelengthSource
        self._rng = rng
        self._polarized = polarized
        self._blockSize = blockSize
        self._nRNGSamples = (
            camera.nRNGSamples + wavelengthSource.nRNGSamples + response.nRNGSamples
        )
        if polarized:
            self._nRNGSamples += 3
        if self._nRNGSamples > 0 and rng is None:
            raise ValueError("An RNG is required but none was specified!")
        # allocate queues
        item = PolarizedCameraHitResponseItem if polarized else CameraHitResponseItem
        self._queue = IOQueue(item, batchSize, mode="retrieve", skipCounter=True)

        # prepare response
        c = TraceConfig(batchSize, blockSize, batchSize, 1, 1.0, polarized)
        response.prepare(c)

        # create code if needed
        if code is None:
            preamble = createPreamble(
                BLOCK_SIZE=blockSize,
                CAMERA_QUEUE_POLARIZED=polarized,
                POLARIZATION=polarized,
                QUEUE_SIZE=batchSize,
            )
            headers = {
                "camera.glsl": camera.sourceCode,
                "photon.glsl": wavelengthSource.sourceCode,
                "rng.glsl": "" if rng is None else rng.sourceCode,
                "response.glsl": response.sourceCode,
            }
            code = compileShader("camera.response.sample.glsl", preamble, headers)
        self._code = code
        self._program = hp.Program(self._code)
        # calculate group size
        self._groups = -(batchSize // -blockSize)
        # bind memory
        self._program.bindParams(QueueOut=self._queue.tensor)

    @property
    def batchSize(self) -> int:
        """Number of samples per batch"""
        return self._batchSize

    @property
    def blockSize(self) -> int:
        """Number of samples per workgroup"""
        return self._blockSize

    @property
    def camera(self) -> Camera:
        """Camera creating the samples"""
        return self._camera

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching"""
        return self._code

    @property
    def nRNGSamples(self) -> int:
        """Number of random numbers drawn per sample"""
        return self._nRNGSamples

    @property
    def queue(self) -> IOQueue:
        """Queue holding the sampled items"""
        return self._queue

    @property
    def response(self) -> HitResponse:
        """Response processing the sampled hits"""
        return self._response

    @property
    def rng(self) -> RNG | None:
        """Random number generator used"""
        return self._rng

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source from which wavelengths are sampled"""
        return self._photons

    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all pipeline stages involved with this tracer in the
        correct order suitable for creating a pipeline.
        """
        return [
            *([self.rng] if self.rng is not None else []),
            self.wavelengthSource,
            self.camera,
            self,
            self.response,
        ]

    def run(self, i):
        self._bindParams(self._program, i)
        self.camera.bindParams(self._program, i)
        self.wavelengthSource.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        if self.rng is not None:
            self.rng.bindParams(self._program, i)
        return [self._program.dispatch(self._groups), *self.queue.run(i)]


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
    def capacity(self) -> int:
        """Maximum number of histograms"""
        # since input tensor may have been provided externally, we need to calculate it
        return self.histIn.size_bytes // (4 * self.nBins)

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
        cmds = [
            hp.flushMemory(),
            self._program.dispatch(self._groups),
        ]
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
        from the corresponding tracer during preparation and config updates.
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
        super().__init__(
            {"ResponseParams": self.Params},
            {"normalization"},
            nRNGSamples=response.nRNGSamples,
        )
        # save params
        self._response = response
        self.setParams(t0=t0, binSize=binSize)
        self._nBins = nBins
        self._normalization = normalization
        self._requestedNorm = 1.0  # keeps track of norm passed through TraceConfigs
        self._retrieve = retrieve
        self._reduceBlockSize = reduceBlockSize
        self._updateResponse = updateResponse
        self._nHistMax = 0  # keeps track of worst case between prepare() calls
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
    def normalization(self, value: float | None) -> None:
        self._normalization = value
        if self._reducer is not None and value is not None:
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
        self._requestedNorm = config.normalization
        nHistMax = -(config.capacity // -config.blockSize)
        if self._reducer is None:
            self._nHistMax = max(self._nHistMax, nHistMax)
        elif nHistMax > self._reducer.capacity:
            raise RuntimeError(
                f"Tracer may produce up to {nHistMax} intermediate histograms, "
                f"but the response can only fit {self._reducer.capacity}"
            )

    def updateConfig(self, config: TraceConfig) -> None:
        if self._reducer is None:
            self._requestedNorm = config.normalization
        else:
            self._reducer.nHist = -(config.batchSize // -config.blockSize)
            if self.normalization is None:
                self._reducer.normalization = config.normalization

    def _initReducer(self) -> HistogramReducer:
        if self.normalization is not None:
            norm = self.normalization
        else:
            norm = self._requestedNorm
        # return instead of just assigning to make type checker happy...
        return HistogramReducer(
            nBins=self.nBins,
            nHist=self._nHistMax,
            retrieve=self.retrieve,
            normalization=norm,
            blockSize=self._reduceBlockSize,
        )

    def _bindParams(self, program: hp.Program, i: int) -> None:
        super()._bindParams(program, i)
        # connect response and reducer
        if self._reducer is None:
            self._reducer = self._initReducer()
        program.bindParams(HistogramOut=self._reducer.histIn)

    def update(self, i: int) -> None:
        if self._reducer is None:
            self._reducer = self._initReducer()
        super().update(i)
        self._reducer.update(i)
        if self._updateResponse:
            self.response.update(i)

    def run(self, i: int) -> list[hp.Command]:
        if self._reducer is None:
            raise RuntimeError("Response has not been prepared")
        return self._reducer.run(i)


class KernelHistogramHitResponse(HitResponse):
    """
    Response function producing a histogram using the values produced by the
    given `ValueResponse`. Unlike `HistogramHitResponse`, however, each hit
    value gets smeared by a kernel function and thus may affect multiple bins.
    This is equivalent to binning a kernel density estimation.

    Currently only Gaussian kernel is supported.

    Parameters
    ----------
    response: ValueResponse
        Value response function processing hits.
    nBins: int, default=100
        Number of bins in the histogram.
    t0: float, default=0ns
        First bin edge, i.e. the minimum time value a sample has to have to get
        included.
    binSize: float, default=1.0ns
        Size of a single bin in units of time
    kernelBandwidth: float, default=1.0ns
        Bandwidth of the kernel. Corresponds to the standard deviation for
        gaussian kernel.
    kernelSupport: float, default=3.0ns
        Limits the range the kernel affects to +/- the support.
    normalization: float | None, default=None
        Common factor each bin gets multiplied with. If `None`, uses the value
        reported from the tracer during the preparation step and config updates.
    retrieve: bool, default=True
        Whether to retrieve the final histogram, i.e. copying it back to CPU.
    reduceBlockSize: int, default=128
        Workgroup size of the reduction stage.
    updateResponse: bool, default=True
        If True and this stage is requested to update e.g. by a pipeline, it
        will cause the response function to also update.

    Stage Parameters
    ----------------
    t0: float
        First bin edge, i.e. the minimum time value a sample has to have to get
        included
    binSize: float
        Size of a single bin in unit of time
        kernelBandwidth: float
        Bandwidth of the kernel. Corresponds to the standard deviation for
        gaussian kernel.
    kernelSupport: float
        Limits the range the kernel affects to +/- the support.
    normalization
        Common factor each bin gets multiplied with

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

    name = "Kernel Histogram Hit Response"

    # lazily load source code
    _sourceCode = ShaderLoader("response.histogram.kernel.glsl")

    class Params(Structure):
        _fields_ = [
            ("t0", c_float),
            ("binSize", c_float),
            ("kernelBandwidth", c_float),
            ("kernelSupport", c_float),
        ]

    def __init__(
        self,
        response: ValueResponse,
        *,
        nBins: int = 100,
        t0: float = 0.0 * u.ns,
        binSize: float = 1.0 * u.ns,
        kernelBandwidth: float = 1.0 * u.ns,
        kernelSupport: float = 3.0 * u.ns,
        normalization: float | None = None,
        retrieve: bool = True,
        reduceBlockSize: int = 128,
        updateResponse: bool = True,
    ) -> None:
        super().__init__(
            {"ResponseParams": self.Params},
            {"normalization"},
            nRNGSamples=response.nRNGSamples,
        )
        # save params
        self._response = response
        self._nBins = nBins
        self._normalization = normalization
        self._requestedNorm = 1.0  # keeps track of norm passed through TraceConfigs
        self._retrieve = retrieve
        self._reduceBlockSize = reduceBlockSize
        self._updateResponse = updateResponse
        self.setParams(
            t0=t0,
            binSize=binSize,
            kernelBandwidth=kernelBandwidth,
            kernelSupport=kernelSupport,
        )
        self._nHistMax = 0  # keeps track of worst case between prepare() calls
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
    def normalization(self, value: float | None) -> None:
        self._normalization = value
        if self._reducer is not None and value is not None:
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
        self._requestedNorm = config.normalization
        nHistMax = -(config.capacity // -config.blockSize)
        if self._reducer is None:
            self._nHistMax = max(self._nHistMax, nHistMax)
        elif nHistMax > self._reducer.capacity:
            raise RuntimeError(
                f"Tracer may produce up to {nHistMax} intermediate histograms, "
                f"but the response can only fit {self._reducer.capacity}"
            )

    def updateConfig(self, config: TraceConfig) -> None:
        if self._reducer is None:
            self._requestedNorm = config.normalization
        else:
            self._reducer.nHist = -(config.batchSize // -config.blockSize)
            if self.normalization is None:
                self._reducer.normalization = config.normalization

    def _initReducer(self) -> HistogramReducer:
        if self.normalization is not None:
            norm = self.normalization
        else:
            norm = self._requestedNorm
        # return instead of just assigning to make type checker happy...
        return HistogramReducer(
            nBins=self.nBins,
            nHist=self._nHistMax,
            retrieve=self.retrieve,
            normalization=norm,
            blockSize=self._reduceBlockSize,
        )

    def _bindParams(self, program: hp.Program, i: int) -> None:
        super()._bindParams(program, i)
        # connect response and reducer
        if self._reducer is None:
            self._reducer = self._initReducer()
        program.bindParams(HistogramOut=self._reducer.histIn)

    def update(self, i: int) -> None:
        if self._reducer is None:
            self._reducer = self._initReducer()
        super().update(i)
        self._reducer.update(i)
        if self._updateResponse:
            self.response.update(i)

    def run(self, i: int) -> list[hp.Command]:
        if self._reducer is None:
            raise RuntimeError("Response has not been prepared")
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
