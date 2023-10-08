import importlib.resources
import hephaistos as hp
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
