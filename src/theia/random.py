import importlib.resources
import hephaistos as hp
import numpy as np

from ctypes import Structure, c_uint32
from os import urandom

from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from theia.util import compileShader, loadShader

from typing import Dict, List, Optional, Set, Tuple, Type


class RNG(SourceCodeMixin):
    """
    Base class for random number generator used in pipeline stages.
    Running this as a stage inside a pipeline updates following stages.
    """

    name = "RNG"

    def __init__(
        self,
        params: Dict[str, Type[Structure]] = {},
        extras: Set[str] = set(),
    ) -> None:
        super().__init__(params, extras)


class RNGBufferSink(PipelineStage):
    """
    Helper class for filling a tensor with random numbers from a generator.

    Samples are stored in tensor with consecutive numbers being in consecutive
    streams.

    Parameters
    ----------
    generator: RNG
        The underlying random number generator samples are drawn from
    streams: int
        Total number of parallel streams
    samples: int
        Number of samples drawn per stream
    baseStream: int, default=0
        Index of first stream
    baseCount: int, default=0
        Offset into each stream
    blockSize: (float, float, float), default=(32, 4, 16)
        Dimension of a single work group in the shape of
        (STREAMS, BATCHES, DRAWS PER BATCH)
        Can be used to tune the performance to a specific device.
    code: bytes | None, default=None
        Compiled source code. If `None`, the byte code get's compiled from
        source. Note, that the compiled code is not checked. You must ensure
        it matches the configuration.

    Stage Parameters
    ----------------
    baseStream: int, default=0
        Index of first stream
    baseCount: int, default=0
        Offset into each stream

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
    def blockSize(self) -> Tuple[float, float, float]:
        """Block size used by the shader: (streams, batches, draws per invocation)"""
        return self._blockSize

    @property
    def capacity(self) -> int:
        """Number of total samples stored in the tensor"""
        return self._capacity

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching"""
        return self._code

    @property
    def generator(self) -> RNG:
        """The underlying random number generator samples are drawn from"""
        return self._generator

    @property
    def streams(self) -> int:
        """Total number of parallel streams"""
        return self.getParam("_streams")

    @property
    def samples(self) -> int:
        """Number of samples drawn per stream"""
        return self.getParam("_samples")

    @property
    def tensor(self) -> hp.FloatTensor:
        """Tensor holding the drawn samples"""
        return self._tensor

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

    Stage Parameters
    ----------------
    key: int | None, default=None
        Base key used by the random number generator. Consecutive streams
        increment the key by one
    offset: int, default=0
        Offset into each stream. Can be used to advance the generator.
    """

    class PhiloxParams(Structure):
        _fields_ = [("key", Key), ("offset", Counter)]

    # cache for source code, lazily loaded (not need if byte code was cached)
    source_code = None

    def __init__(self, *, key: Optional[int] = None, offset: int = 0) -> None:
        super().__init__({"PhiloxParams": self.PhiloxParams})
        # save params
        if key is None:
            key = urandom(8)
        self.setParams(key=key, offset=offset)

    @property
    def sourceCode(self) -> str:
        if PhiloxRNG.source_code is None:
            PhiloxRNG.source_code = loadShader("random.philox.glsl")
        return PhiloxRNG.source_code


class SobolQRNG(RNG):
    """
    Sobol sequence optionally scrambled using fast Owen scrambling.

    Parameters
    ----------
    seed: int | None, default=None
        Base seed used for scrambling. If None, asks the system to create a
        random one. Ignored if scrambled is False.
    offset: int, default=0
        Offset into the sequence. Can be used to split draws into multiple runs.
        Should not be used to skip samples altogether.
    scramble: bool, default=True
        Wether to scramble the sequence

    Stage Parameters
    ----------------
    seed: int | None, default=None
        Base seed used for scrambling. If None, asks the system to create a
        random one. Ignored if scrambled is False.
    offset: int, default=0
        Offset into the sequence. Can be used to split draws into multiple runs.
        Should not be used to skip samples altogether.

    Note
    ----
    Only supports drawing samples up to dimension 1024
    """

    class SobolParams(Structure):
        _fields_ = [("seed", c_uint32), ("offset", c_uint32)]

    # cache for source source, lazily loaded (not need if byte code was cached)
    source_code = None

    def __init__(
        self, *, seed: Optional[int] = None, offset: int = 0, scrambled: bool = True
    ) -> None:
        super().__init__({"SobolParams": self.SobolParams})
        # save params
        self._scrambled = scrambled
        if seed is None:
            seed = urandom(4)
        self.setParams(seed=seed, offset=offset)

        # load sobol matrices
        path = importlib.resources.files("theia").joinpath("data/sobolmatrices.npy")
        matrices = np.load(path)
        # upload to device
        self._matrices = hp.IntTensor(matrices.flatten())
        # while it would make sense to cache the matrices between instances,
        # we'd have to somehow detect when the device was destroyed and recreate
        # the tensor. Since it's not that big, this is far easier and safer

    @property
    def scrambled(self) -> bool:
        """Wether the sequence is scrambled"""
        return self._scrambled

    @property
    def sourceCode(self) -> str:
        if SobolQRNG.source_code is None:
            SobolQRNG.source_code = loadShader("random.sobol.glsl")
        # add preamble if needed
        if self.scrambled:
            # add define to disable scrambling if needed
            # multiple #define are allowed so dont sweat about putting it before
            # the include guard
            preamble = "#define _SOBOL_NO_SCRAMBLE\n\n"
            return preamble + SobolQRNG.source_code
        else:
            return SobolQRNG.source_code

    def bindParams(self, program: hp.Program, i: int) -> None:
        super().bindParams(program, i)
        program.bindParams(SobolMatrices=self._matrices)
