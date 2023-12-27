import hephaistos as hp

from ctypes import Structure, c_uint32
from os import urandom

from hephaistos.pipeline import PipelineStage
from theia.util import compileShader, loadShader

from typing import Dict, List, Optional, Set, Tuple, Type


class RNG(PipelineStage):
    """
    Base class for random number generator used in pipeline stages.
    Running this as a stage inside a pipeline updates following stages.
    """

    name = "RNG"

    def __init__(
        self,
        sourceCode: str,
        params: Dict[str, Type[Structure]] = {},
        extras: Set[str] = set(),
        *,
        alignment: int = 1,
    ) -> None:
        super().__init__(params, extras)
        self._sourceCode = sourceCode
        self._alignment = alignment

    @property
    def alignment(self) -> int:
        """Alignment each stream (i.e. idx offset) should have for optimal speed"""
        return self._alignment

    @property
    def sourceCode(self) -> str:
        """
        Source to be used by the pipeline stage.
        Can be shared among multiple stages.
        """
        return self._sourceCode

    def bindParams(self, program: hp.Program, i: int) -> None:
        """
        Binds params used by the generator to the program for the given configuration
        """
        self._bindParams(program, i)

    def run(self, i: int) -> List[hp.Command]:
        # RNG are most likely config only
        return []


class RNGBufferSink(PipelineStage):
    """
    Helper class for filling a tensor with random numbers from a generator.

    Samples are stored in tensor with consecutive numbers being in consecutive
    streams.

    Example
    -------
    >>> import numpy as np
    >>> import theia.random
    >>> from hephaistos.pipeline import RetrieveTensorStage, runPipeline
    >>> gen = theia.random.PhiloxRNG()
    >>> samples, streams = 1024, 8192
    >>> sink = theia.random.RNGBufferSink(gen, streams, samples)
    >>> ret = RetrieveTensorStage(sink.tensor)
    >>> runPipeline([gen, sink, ret])
    >>> samples = ret.view(0).reshape((samples, streams))
    """

    name = "RNG Sink"

    class Params(Structure):
        _fields_ = [
            ("baseStream", c_uint32),
            ("baseCount", c_uint32),
            ("_streams", c_uint32),
            ("_samples", c_uint32),
        ]

    def __init__(
        self,
        generator: RNG,
        streams: int = 256,
        samples: int = 256,
        *,
        baseStream: int = 0,
        baseCount: int = 0,
        blockSize: Tuple[int, int, int] = (32, 4, 16),
        code: Optional[bytes] = None,
    ) -> None:
        super().__init__({"Params": self.Params})
        # save params
        capacity = streams * samples
        self._capacity = capacity
        self._generator = generator
        self._blockSize = blockSize
        self.setParams(
            baseStream=baseStream,
            baseCount=baseCount,
            _streams=streams,
            _samples=samples,
        )

        # create code if needed
        if code is None:
            preamble = ""
            preamble += f"#define PARALLEL_STREAMS {blockSize[0]}\n"
            preamble += f"#define BATCH_SIZE {blockSize[1]}\n"
            preamble += f"#define DRAWS {blockSize[2]}\n\n"
            headers = {"rng.glsl": generator.sourceCode}
            code = compileShader("random.sink.glsl", preamble, headers)
        self._code = code
        self._program = hp.Program(self._code)
        # calculate dispatch size (ceil division)
        self._dispatchSize = (
            -(streams // -blockSize[0]),
            -(samples // -(blockSize[1] * blockSize[2])),
        )

        # allocate memory
        self._tensor = hp.FloatTensor(capacity)
        self._program.bindParams(RngSink=self._tensor)

    @property
    def streams(self) -> int:
        """Total number of streams"""
        return self.getParam("_streams")

    @property
    def samples(self) -> int:
        """Number of samples drawn per stream"""
        return self.getParam("_samples")

    @property
    def capacity(self) -> int:
        """Number of total samples stored in the tensor"""
        return self._capacity

    @property
    def tensor(self) -> hp.FloatTensor:
        """Tensor holding the drawn samples"""
        return self._tensor

    @property
    def generator(self) -> RNG:
        """The underlying random number generator samples are drawn from"""
        return self._generator

    @property
    def blockSize(self) -> Tuple[float, float, float]:
        """Block size used by the shader: (streams, batches, draws per invocation)"""
        return self._blockSize

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        self.generator.bindParams(self._program, i)
        return [self._program.dispatch(*self._dispatchSize)]


class Key(Structure):
    _fields_ = [("lo", c_uint32), ("hi", c_uint32)]

    @property
    def value(self) -> int:
        return self.lo + (self.hi << 32)

    @value.setter
    def value(self, value: int) -> None:
        self.lo = value & 0xFFFFFFFF
        self.hi = value >> 32 & 0xFFFFFFFF


class Counter(Structure):
    _fields_ = [("word", c_uint32 * 4)]

    @property
    def value(self) -> int:
        return sum(self.word[i] << (32 * i) for i in range(4))

    @value.setter
    def value(self, value: int) -> None:
        for i in range(4):
            self.word[i] = value >> (32 * i) & 0xFFFFFFFF


class PhiloxRNG(RNG):
    """
    Philox 4x32 RNG

    Parameters
    ----------
    key: int | None, default=None
        Base key used by the random number generator. Consecutive streams
        increment the key by one
    offset: int, default=0
        Offset into each stream. Can be used to advance the generator.
    """

    class PhiloxParams(Structure):
        _fields_ = [("key", Key), ("offset", Counter)]

    def __init__(self, *, key: Optional[int] = None, offset: int = 0) -> None:
        super().__init__(
            loadShader("random.philox.glsl"),
            {"PhiloxParams": self.PhiloxParams},
            alignment=4,
        )

        # save params
        if key is None:
            key = urandom(8)
        self.setParams(key=key, offset=offset)
