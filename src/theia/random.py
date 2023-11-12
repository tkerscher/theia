import hephaistos as hp

from ctypes import Structure, c_uint32
from hephaistos.glsl import uvec2, uvec4
from theia.scheduler import CachedPipelineStage
from os import urandom
from typing import Dict, List, Optional, Tuple, Type
from .util import (
    compileShader,
    loadShader,
    intToUvec4,
    uvec4ToInt,
    packUint64,
    unpackUint64,
)


class RNG(CachedPipelineStage):
    """
    Base class for random number generator used in pipeline stages.
    Running this as a stage inside a pipeline updates following stages.
    """

    def __init__(
        self,
        sourceCode: str,
        params: Dict[str, Type[Structure]] = {},
        *,
        nConfigs: int = 2,
        alignment: int = 1,
    ) -> None:
        super().__init__(params, nConfigs)
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
        self._bindConfigs(program, i)

    def run(self, i: int) -> List[hp.Command]:
        # RNG are most likely config only
        return []


class RNGBufferSink(CachedPipelineStage):
    """
    Helper class for filling a tensor with random numbers from a generator
    """

    class Params(Structure):
        _fields_ = [
            ("baseStream", c_uint32),
            ("baseCount", c_uint32),
            ("streams", c_uint32),
            ("samples", c_uint32),
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
        super().__init__({"Params": self.Params}, generator.nConfigs)
        # save params
        capacity = streams * samples
        self._setParam("Params", "streams", streams)
        self._setParam("Params", "samples", samples)
        self._capacity = capacity
        self._generator = generator
        self.baseStream = baseStream
        self.baseCount = baseCount
        self._blockSize = blockSize

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
        return self._getParam("Params", "streams")

    @property
    def samples(self) -> int:
        """Number of samples drawn per stream"""
        return self._getParam("Params", "samples")

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

    @property
    def baseStream(self) -> int:
        """Offset added to the streams"""
        return self._getParam("Params", "baseStream")

    @baseStream.setter
    def baseStream(self, value: int) -> None:
        self._setParam("Params", "baseStream", value)

    @property
    def baseCount(self) -> int:
        """Offset into a single stream"""
        return self._getParam("Params", "baseCount")

    @baseCount.setter
    def baseCount(self, value: int) -> None:
        self._setParam("Params", "baseCount", value)

    def update(self, i: int) -> None:
        super().update(i)

    def run(self, i: int) -> List[hp.Command]:
        self._bindConfigs(self._program, i)
        self.generator.bindParams(self._program, i)
        return [self._program.dispatch(*self._dispatchSize)]


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
        _fields_ = [("key", uvec2), ("baseCount", uvec4)]

    def __init__(
        self, *, key: Optional[int] = None, offset: int = 0, nConfigs: int = 2
    ) -> None:
        super().__init__(
            loadShader("random.philox.glsl"),
            {"PhiloxParams": self.PhiloxParams},
            nConfigs=nConfigs,
            alignment=4,
        )

        # save params
        if key is None:
            key = urandom(8)
        self.key = key
        self.offset = offset

    @property
    def key(self) -> int:
        """Base key used for Philox. Consecutive streams differ in their keys by one."""
        return unpackUint64(self._getParam("PhiloxParams", "baseCount"))

    @key.setter
    def key(self, value: int) -> None:
        self._setParam("PhiloxParams", "key", packUint64(value))

    @property
    def offset(self) -> int:
        """Offset into each stream. Can be used to advance the generator."""
        return uvec4ToInt(self._getParam("PhiloxParams", "baseCount"))

    @offset.setter
    def offset(self, value: int) -> None:
        self._setParam("PhiloxParams", "baseCount", intToUvec4(value))
