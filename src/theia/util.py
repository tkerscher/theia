from __future__ import annotations

import numpy as np
import importlib.resources

from ctypes import _SimpleCData, Array, Structure, c_uint32
from hephaistos.glsl import uvec4, uvec2
from numpy.ctypeslib import as_array

from numpy.typing import NDArray
from typing import Sequence


class classproperty:
    """class property decorator"""

    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


def intersectRange(
    a: tuple[float, float] | None,
    b: tuple[float, float] | None,
    /,
) -> tuple[float, float]:
    """Returns the intersection of two ranges"""
    if a is None and b is None:
        raise ValueError("Can not combine `None` and `None`!")
    a_min, a_max = (float("-inf"), float("inf")) if a is None else a
    b_min, b_max = (float("-inf"), float("inf")) if b is None else b
    c_min, c_max = max(a_min, b_min), min(a_max, b_max)
    if not c_min <= c_max:  # this way we handle NaNs
        raise ValueError("Ranges do not overlap!")
    return (c_min, c_max)


def loadCSV(file: str, *, delimiter: str | None = ",", skiprows: int = 1) -> NDArray:
    return np.loadtxt(
        str(importlib.resources.files("theia").joinpath(f"data/{file}")),
        delimiter=delimiter,
        skiprows=skiprows,
    )


def createCType(name: str, fields: Sequence[_SimpleCData | Array]) -> type[Structure]:
    """Util function for dynamically creating ctypes structures"""
    return type(name, (Structure,), {"_fields_": fields})


def viewSoA(address: int, item: type[Structure], count: int) -> NDArray:
    """
    Returns a structured numpy array as a view of the structure of array saved
    at the given memory address.
    """

    class SoA(Structure):
        _fields_ = [(name, t * count) for name, t in item._fields_]

    return as_array(SoA.from_address(address))


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


def packUint64(value: int) -> uvec2:
    """Packs a 64bit unsigned integer into a 2D 32bit unsigned integer vector"""
    return uvec2(x=c_uint32(value & 0xFFFFFFFF), y=c_uint32(value >> 32 & 0xFFFFFFFF))


def unpackUint64(value: uvec2) -> int:
    """Unpacks a packed 64 bit unsigned integer"""
    return value.x + (value.y << 32)
