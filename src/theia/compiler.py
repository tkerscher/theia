from __future__ import annotations

import importlib.resources
import hephaistos as hp

from theia.device import getEnabledRayTracingFeatures

from importlib.resources.abc import Traversable
from typing import Any


__all__ = [
    "addIncludeDir",
    "compileShader",
    "createPreamble",
    "findShader",
    "getCompiler",
    "getPreamble",
    "loadShader",
]


def __dir__():
    return __all__


# global state
_shaderDir = importlib.resources.files("theia").joinpath("shader/")
_includeDirs: list[Traversable] = [_shaderDir]
_compiler = hp.Compiler()
_compiler.addIncludeDir(str(_shaderDir))


def getCompiler() -> hp.Compiler:
    """Returns the compiler used internally by theia"""
    return _compiler


def addIncludeDir(path: Traversable) -> None:
    """adds the given path to the list of include dirs searched by theia"""
    _includeDirs.append(path)
    _compiler.addIncludeDir(str(path))


def findShader(file: str) -> Traversable:
    """
    Tries to find the given file in the current list of include directories and
    returns the path under which it can be found.
    Raises `FileNotFoundError` if the file cannot be found.
    """
    for dir in _includeDirs:
        candidate_path = dir.joinpath(file)
        if candidate_path.is_file():
            return candidate_path

    # nothing found
    raise FileNotFoundError()


def loadShader(file: str) -> str:
    """
    Tries to find the given shader file in the list of include dirs and returns
    the source code it contains. Raises `FileNotFoundError` if the file could
    not be found.
    """
    with findShader(file).open("r") as f:
        return f.read()


# load preamble. we will always need it
PREAMBLE = loadShader("preamble.glsl")


def getPreamble() -> str:
    """Returns preamble preadded to all shader code before compiling"""
    # build preamble
    preamble = str(PREAMBLE)  # copy for modification
    # the device may have changed between calls so it's safer to just rebuilt
    # it every time we need it.
    # enable supported ray tracing features
    rt_features = getEnabledRayTracingFeatures()
    if rt_features.query:
        preamble += "#extension GL_EXT_ray_query : require\n"
    if rt_features.pipeline:
        preamble += "#extension GL_EXT_ray_tracing : require\n"
    if rt_features.positionFetch:
        preamble += "#extension GL_EXT_ray_tracing_position_fetch : require\n"
        preamble += "#define POSITION_FETCH_ENABLED 1\n"
    # done
    return preamble


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


def compileShader(
    file: str,
    preamble: str = "",
    headers: dict[str, str] = {},
    *,
    stage: hp.ShaderStage = hp.ShaderStage.COMPUTE,
) -> bytes:
    """
    Compiles the shader stored in the given file. Must be stored in one of the
    include directories currently registered.

    Parameters
    ----------
    file: str
        Path to the file containing the shader code
    preamble: str, default=""
        Additional code prepended to the source code
    headers: { str, str }, default={}
        Map of runtime header files mapping include name to source code
    stage: hp.ShaderStage, default=COMPUTE
        For which stage to compile the code

    Returns
    -------
    bytecode: bytes
        Compiled SPIR-V byte code
    """
    # a common error is to screw up the headers map
    # quick sanity check to create a more meaningfull error
    for header, content in headers.items():
        if not isinstance(content, str):
            raise ValueError(f'Header "{header}" does not contain a string!')

    # combine all strings into a single source code for the compiler.
    # add line macros to make potential errors more meaningful
    code = "\n".join(
        [
            getPreamble(),
            f'#line 1 "preamble.glsl"',
            preamble,
            f'#line 1 "{file}"',
            loadShader(file),
        ]
    )
    # finally ask compiler to compile the code
    return getCompiler().compile(code, headers, stage=stage)
