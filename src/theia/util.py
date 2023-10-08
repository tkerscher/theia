import importlib.resources
import hephaistos as hp
from ctypes import c_uint32
from hephaistos.glsl import uvec4
from os.path import join


def getCompiler() -> hp.Compiler:
    """Returns a lazy singleton compiler with lib shader code include dir"""
    if not hasattr(getCompiler, "_compiler"):
        shader_dir = str(importlib.resources.files("theia").joinpath("shader"))
        compiler = hp.Compiler()
        compiler.addIncludeDir(shader_dir)
        getCompiler._compiler = compiler
    return getCompiler._compiler


def compileShader(file: str) -> bytes:
    """Compiles the given shader code stored inside the libs shader folder"""
    path = str(importlib.resources.files("theia").joinpath(f"shader/{file}"))
    source = None
    with open(path, "r") as file:
        source = file.read()
    return getCompiler().compile(source)


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
