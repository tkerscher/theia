import hephaistos as hp
import importlib.resources
import numpy as np
from ctypes import Structure, c_float, c_uint32, sizeof
from hephaistos.glsl import uvec2, uvec4, vec3
from math import ceil
from os import urandom
from typing import Optional
from .util import compileShader


def uvec4ToInt(value: uvec4) -> int:
    """Transforms a uvec4 value to int"""
    return value.x + (value.y << 32) + (value.z << 64) + (value.w << 96)


def intToUvec4(value: int) -> uvec4:
    """Transforms an int to an uvec4"""
    return uvec4(
        x=c_uint32(value & 0xFFFFFFFF),
        y=c_uint32(value >> 32 & 0xFFFFFFFF),
        z=c_uint32(value >> 64 & 0xFFFFFFFF),
        w=c_uint32(value >> 96 & 0xFFFFFFFF),
    )


class PhiloxRNG:
    """
    Program for filling a tensor with random numbers using the Philox 4x32
    algorithm. Always creates 4 floats per invocation
    """

    # cache the shader code to prevent multiple compilation
    _code = None

    class Push(Structure):
        _fields_ = [("baseKey", uvec2), ("baseCounter", uvec4), ("stride", c_uint32)]

    def __init__(
        self,
        streams: int,
        batch: int,
        *,
        key: Optional[int] = None,
        startCount: int = 0
    ) -> None:
        # check if streams is a multiple of 32 (local size)
        # TODO: ease this restriction
        if streams % 32:
            raise RuntimeError("streams must be a multiple of 32!")
        self._streams = streams
        # check if batch is multiple of 4
        if batch % 4:
            raise RuntimeError("batch must be a multiple of 4!")
        self._batch = batch
        # generate random key if none provided
        if key is None:
            key = urandom(8)
        mask = 0xFFFFFFFF
        _key = uvec2(x=c_uint32(key & mask), y=c_uint32((key >> 32) & mask))
        _counter = intToUvec4(startCount)
        # save params
        self._push = PhiloxRNG.Push(
            stride=batch // 4, baseKey=_key, baseCounter=_counter
        )
        # create tensor
        self._tensor = hp.FloatTensor(streams * batch)

        # compile shader if needed
        if PhiloxRNG._code is None:
            PhiloxRNG._code = compileShader("philox.buffer.glsl")
        # create program
        self._program = hp.Program(PhiloxRNG._code)
        self._program.bindParams(RNGBuffer=self._tensor)

    @property
    def streams(self) -> int:
        """Number of parallel rng streams"""
        return self._streams

    @property
    def batch(self) -> int:
        """Number of samples to draw for each stream per round"""
        return self._batch

    @property
    def tensor(self) -> hp.FloatTensor:
        """Tensor holding the generated random numbers"""
        return self._tensor

    def dispatchNext(self) -> None:
        """Creates a command for drawing the next batch of random numbers"""
        # dispatch
        cmd = self._program.dispatchPush(
            bytes(self._push), self.streams // 32, self.batch // 4
        )
        # update push
        counter = uvec4ToInt(self._push.baseCounter)
        counter += self.batch
        self._push.baseCounter = intToUvec4(counter)
        # return cmd
        return cmd
