import importlib.resources
import hephaistos as hp
from ctypes import c_uint32, c_uint64
from hephaistos.glsl import uvec4, uvec2
from os.path import join
from typing import Dict


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


def compileShader(
        file: str,
        preamble: str = "",
        headers: Dict[str, str] = {}
    ) -> bytes:
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
    return getCompiler().compile(code, headers)


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
    return uvec2(
        x = c_uint32(value & 0xFFFFFFFF),
        y = c_uint32(value >> 32 & 0xFFFFFFFF)
    )


def unpackUint64(value: uvec2) -> int:
    """Unpacks a packed 64 bit unsigned integer"""
    return value.x + (value.y << 32)
