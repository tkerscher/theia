from __future__ import annotations
import hephaistos as hp

from collections import namedtuple
from ctypes import Structure, c_float, c_uint32
from hephaistos.glsl import vec3, uvec2
from numpy.ctypeslib import as_array
from numpy.typing import NDArray
from typing import Optional, Union, Type
from .util import compileShader, packUint64


ItemSize = namedtuple("ItemSize", ["const", "perPhoton"])
"""Tuple describing the size of an item inside a single queue"""


def createQueue(item: ItemSize, nPhotons: int, n: int) -> hp.ByteTensor:
    """Creates a tensor with enough space to hold a queue of n items"""
    itemSize = item.const + item.perPhoton * nPhotons
    size = 4 + itemSize * n
    return hp.ByteTensor(size)
