from __future__ import annotations

import importlib.resources
import hephaistos as hp

from ctypes import Structure, c_uint32
from hephaistos.glsl import uvec4, uvec2
from numpy.ctypeslib import as_array

from numpy.typing import NDArray
from typing import Any, Dict, Type


def viewSoA(address: int, item: Type[Structure], count: int) -> NDArray:
    """
    Returns a structured numpy array as a view of the structure of array saved
    at the given memory address.
    """

    class SoA(Structure):
        _fields_ = [(name, t * count) for name, t in item._fields_]

    return as_array(SoA.from_address(address))


def getCompiler() -> hp.Compiler:
    """Returns a lazy singleton compiler with lib shader code include dir"""
    if not hasattr(getCompiler, "_compiler"):
        shader_dir = str(importlib.resources.files("theia").joinpath("shader"))
        compiler = hp.Compiler()
        compiler.addIncludeDir(shader_dir)
        getCompiler._compiler = compiler
    return getCompiler._compiler


def getShaderPath(file: str) -> str:
    """Returns the path to the given shader file"""
    return str(importlib.resources.files("theia").joinpath(f"shader/{file}"))


def loadShader(file: str) -> str:
    """Loads the given shader file and returns its source code"""
    with open(getShaderPath(file), "r") as file:
        return file.read()


class ShaderLoader:
    """Descriptor for lazily loading shader code from the shader folder"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.code = None

    def __get__(self, obj, objtype=None) -> str:
        if obj is None:
            # accessed from class
            return self
        if self.code is None:
            self.code = loadShader(self.path)
        return self.code


def compileShader(file: str, preamble: str = "", headers: Dict[str, str] = {}) -> bytes:
    """
    Compiles the given shader code stored inside the libs shader folder.

    Parameters
    ----------
    file: str
        path to file containing the shader source code
    preamble: str, default=""
        text to prepend the source code
    headers: { str: str }, default={}
        map of runtime header files mapping file name to source code
    """
    code = preamble + "\n" + loadShader(file)
    try:
        return getCompiler().compile(code, headers)
    except RuntimeError as err:
        # Add preamble size to make line numbers useful
        newTxt = f"(Preamble size: {preamble.count("\n") + 1})\n" + str(err)
        raise RuntimeError(newTxt) from err


def createPreamble(**macros: Any) -> str:
    """
    Creates preamble containing macros defining the values as defined in the
    provided dictionary. If macro is of type bool, defines it without value
    if and only if it is True. Other types are used as value for the macro.
    """
    preamble = ""
    for macro, value in macros.items():
        if type(value) is not bool:
            preamble += f"#define {macro} {value}\n"
        elif value:
            preamble += f"#define {macro}\n"
        # ignore macros defined with value False
    return preamble


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
