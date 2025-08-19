from __future__ import annotations

import hephaistos as hp
import numpy as np

from ctypes import Structure, c_uint32
from os import urandom
import warnings

from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from theia.util import ShaderLoader, compileShader, createPreamble


__all__ = [
    "Counter",
    "Key",
    "PhiloxRNG",
    "RNG",
    "RNGBufferSink",
    "SobolQRNG",
]


def __dir__():
    return __all__


class RNG(SourceCodeMixin):
    """
    Base class for random number generator used in pipeline stages.
    Running this as a stage inside a pipeline updates following stages.
    """

    name = "RNG"

    def __init__(
        self,
        params: dict[str, type[Structure]] = {},
        extras: set[str] = set(),
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
    sampleDim: int, default=1
        Dimensionality of the drawn samples. Currently only one and two
        dimensions supported
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
        sampleDim: int = 1,
        blockSize: tuple[int, int, int] = (32, 4, 16),
        code: bytes | None = None,
    ) -> None:
        super().__init__({"Params": self.Params})
        # check params
        if not 1 <= sampleDim <= 2:
            raise ValueError("Unsupported sample dimension requested!")
        # save params
        capacity = streams * samples * sampleDim
        self._capacity = capacity
        self._generator = generator
        self._blockSize = blockSize
        self._shape = (streams, samples, sampleDim)
        self.setParams(
            baseStream=baseStream,
            baseCount=baseCount,
            _streams=streams,
            _samples=samples,
        )

        # create code if needed
        if code is None:
            preamble = createPreamble(
                BATCH_SIZE=blockSize[1],
                DRAWS=blockSize[2],
                SAMPLE_DIM=sampleDim,
                PARALLEL_STREAMS=blockSize[0],
            )
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
    def blockSize(self) -> tuple[float, float, float]:
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
    def shape(self) -> tuple[float, float, float]:
        """Shape of the sample buffer"""
        return self._shape

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

    def run(self, i: int) -> list[hp.Command]:
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
    autoAdvance: int, default=0
        Amount the offset gets incremented with each update call.

    Stage Parameters
    ----------------
    key: int | None, default=None
        Base key used by the random number generator. Consecutive streams
        increment the key by one
    offset: int, default=0
        Offset into each stream. Can be used to advance the generator.
    autoAdvance: int, default=0
        Amount the offset gets incremented with each update call.
    """

    class PhiloxParams(Structure):
        _fields_ = [("key", Key), ("offset", Counter)]

    def __init__(
        self, *, key: int | None = None, offset: int = 0, autoAdvance: int = 0
    ) -> None:
        super().__init__({"PhiloxParams": self.PhiloxParams}, {"autoAdvance"})
        # save params
        if key is None:
            key = int.from_bytes(urandom(8))
            warnings.warn(f"Random RNG key generated: 0x{key:08X}")
        self.setParams(key=key, offset=offset, autoAdvance=autoAdvance)

    # sourceCode via descriptor
    sourceCode = ShaderLoader("random.philox.glsl")

    @property
    def autoAdvance(self) -> int:
        """Amount the offset gets incremented with each update call."""
        return self._autoAdvance

    @autoAdvance.setter
    def autoAdvance(self, value: int) -> int:
        self._autoAdvance = value

    def _finishParams(self, i: int) -> None:
        if self.autoAdvance != 0:
            offset = self.getParam("offset")
            self.setParam("offset", offset + self.autoAdvance)
        super()._finishParams(i)


class SobolQRNG(RNG):
    """
    Shuffled Owen-scrambled Sobol sampler based on the paper "Practical
    Hash-based Owen Scrambling" by Brent Burley, 2020, Journal of Computer
    Graphics Techniques.

    Parameters
    ----------
    seed: int | None, default=None
        Base seed used for scrambling. If None and `advanceSeed` is provided, a
        seed will be drawn from the host side random number generator, otherwise
        a random seed from the OS will be used.
    offset: int, default=0
        Offset into the sequence. Can be used to split draws into multiple runs.
        Should not be used to skip samples altogether.
    advanceSeed: int | None, default=None
        Seeding for host side random number generator used to alter the device
        side seed between runs. If None, seed will not be altered between runs.

    Stage Parameters
    ----------------
    seed: int
        Base seed used for scrambling.
    offset: int, default=0
        Offset into the sequence. Can be used to split draws into multiple runs.
        Should not be used to skip samples altogether.
    advanceSeed: int | None, default=None
        Seeding for host side random number generator used to alter the device
        side seed between runs. If None, seed will not be altered between runs.
    """

    class SobolParams(Structure):
        _fields_ = [("seed", c_uint32), ("offset", c_uint32)]

    def __init__(
        self,
        *,
        seed: int | None = None,
        offset: int = 0,
        advanceSeed: int | None = None,
    ) -> None:
        super().__init__({"SobolParams": self.SobolParams}, {"advanceSeed"})
        # save params
        if seed is None and advanceSeed is None:
            seed = int.from_bytes(urandom(4))
            warnings.warn(f"Random RNG seed generated: 0x{seed:04X}")
        self.setParams(seed=seed, offset=offset, advanceSeed=advanceSeed)

    sourceCode = ShaderLoader("random.sobol.glsl")

    @property
    def advanceSeed(self) -> int | None:
        """
        Seeding for host side random number generator used to alter the device
        side seed between runs. If None, seed will not be altered between runs.
        """
        return self._advanceSeed

    @advanceSeed.setter
    def advanceSeed(self, value: int | None) -> None:
        self._advanceSeed = value
        self._hostRNG = None if value is None else np.random.default_rng(value)

    def _finishParams(self, i: int) -> None:
        if self._hostRNG is not None:
            seed = int.from_bytes(self._hostRNG.bytes(4))
            self.setParam("seed", seed)
        super()._finishParams(i)
