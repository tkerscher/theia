from __future__ import annotations

import importlib.resources
import hephaistos as hp

from functools import cache

from theia.device import getEnabledAtomics, getEnabledRayTracingFeatures

from collections.abc import Iterable
from importlib.resources.abc import Traversable
from typing import Any, overload


__all__ = [
    "CompilerSession",
    "RayTracingPipelineBuilder",
    "addIncludeDir",
    "compileShader",
    "createPreamble",
    "deduceStage",
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


def createCompilerSession() -> hp.CompilerSession:
    """Creates a new compiler session"""
    return _compiler.createSession()


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


def loadShader(file: str, preamble: str | None = None) -> str:
    """
    Tries to find the given shader file in the list of include dirs and returns
    the source code it contains. Raises `FileNotFoundError` if the file could
    not be found.
    """
    with findShader(file).open("r") as f:
        code = f.read()
    if preamble is None:
        return code
    else:
        return "\n".join([preamble, f'#line 1 "{file}"', code])


@cache
def getPreamble() -> str:
    """Returns preamble preadded to all shader code before compiling"""
    # build preamble
    preamble = loadShader("preamble.glsl")
    # enable extended types
    types = hp.getSupportedTypes()
    if types.float64:
        preamble += (
            "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n"
        )
        preamble += "#define SUPPORTS_DOUBLE 1\n"
    # enable supported atomics
    atomics = getEnabledAtomics()
    if hp.Atomics.BufferFloat64Add in atomics:
        preamble += "#define SUPPORTS_ATOMIC_DOUBLE 1\n"
    if hp.Atomics.BufferInt64 in atomics:
        preamble += "#extension GL_EXT_shader_atomic_int64 : require\n"
        preamble += "#define SUPPORTS_ATOMIC_INT64 1\n"
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
    session: hp.CompilerSession | None = None,
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
    compiler = session or getCompiler()
    return compiler.compile(code, headers, stage=stage)


def deduceStage(file: str) -> hp.ShaderStage:
    """Tries to deduct the shader stage based on the given files name"""
    if file.endswith(".rgen.glsl"):
        return hp.ShaderStage.RAYGEN
    if file.endswith(".rchit.glsl"):
        return hp.ShaderStage.CLOSEST_HIT
    if file.endswith(".rmiss.glsl"):
        return hp.ShaderStage.MISS
    if file.endswith(".rcall.glsl"):
        return hp.ShaderStage.CALLABLE
    if file.endswith(".glsl"):
        return hp.ShaderStage.COMPUTE
    else:
        raise ValueError("Unrecognized file extension")


class CompilerSession:
    """Util class for compiling multiple files within a single session"""

    def __init__(
        self, preamble: str = "", headers: dict[str, str] | None = None
    ) -> None:
        self.session = createCompilerSession()
        self.preamble = preamble
        self.headers: dict[str, str] = headers or {}

    @overload
    def compile(
        self, file: str, extraHeaders: dict[str, str] | None = None
    ) -> bytes: ...
    @overload
    def compile(
        self, file: list[str], extraHeaders: dict[str, str] | None = None
    ) -> list[bytes]: ...

    def compile(
        self,
        file: str | list[str],
        extraHeaders: dict[str, str] | None = None,
    ) -> bytes | list[bytes]:
        """Compiles the given files. Deduces the corresponding stages automatically."""
        if extraHeaders:
            headers = {**extraHeaders, **self.headers}
        else:
            headers = self.headers
        comp = lambda f: compileShader(
            f, self.preamble, headers, stage=deduceStage(f), session=self.session
        )
        if isinstance(file, str):
            return comp(file)
        else:
            return [comp(f) for f in file]


RTShader = hp.RayGenerateShader | hp.RayMissShader | hp.RayHitShader | hp.CallableShader
"""Union of all ray tracing shaders types"""


class RayTracingPipelineBuilder:
    """Utility class for creating ray tracing pipelines"""

    def __init__(self) -> None:
        self._shaders: list[RTShader] = []
        self._sbt: dict[str, list[int | None]] = {}

    def _addShader(self, shader: RTShader | None) -> int | None:
        if shader is None:
            return None
        idx = len(self._shaders)
        self._shaders.append(shader)
        return idx

    def addRayGen(self, code: bytes) -> None:
        """Adds the ray generation shader"""
        if "rayGenShaders" in self._sbt:
            raise RuntimeError("ray gen shaders have already been added")
        idx = self._addShader(hp.RayGenerateShader(code))
        self._sbt["rayGenShaders"] = [idx]

    def addMissShaders(self, code: Iterable[bytes | None]) -> None:
        """adds miss shaders"""
        if "missShaders" in self._sbt:
            raise RuntimeError("miss shaders have already been added")
        f = lambda c: None if c is None else hp.RayMissShader(c)
        idx = [self._addShader(f(c)) for c in code]
        self._sbt["missShaders"] = idx

    def addHitShaders(self, code: Iterable[bytes | None]) -> None:
        """adds hit shaders"""
        if "hitShaders" in self._sbt:
            raise RuntimeError("Hit shaders have already been added")
        f = lambda c: None if c is None else hp.RayHitShader(c)
        idx = [self._addShader(f(c)) for c in code]
        self._sbt["hitShaders"] = idx

    def addCallableShaders(self, code: Iterable[bytes | None]) -> None:
        """adds optional callable shaders"""
        if "callableShaders" in self._sbt:
            raise RuntimeError("Callable shaders have already been added")
        f = lambda c: None if c is None else hp.CallableShader(c)
        idx = [self._addShader(f(c)) for c in code]
        if len(idx) > 0:
            self._sbt["callableShaders"] = idx

    def finish(
        self, *, maxRecursionDepth: int = 1
    ) -> tuple[hp.RayTracingPipeline, dict[str, hp.ShaderBindingTable]]:
        """Builds the ray tracing pipeline and corresponding SBT"""
        # check everthing was added
        if "rayGenShaders" not in self._sbt:
            raise RuntimeError("Ray gen shaders missing")
        if "missShaders" not in self._sbt:
            raise RuntimeError("Miss shaders missing")
        if "hitShaders" not in self._sbt:
            raise RuntimeError("Hit shaders missing")
        # callable shaders are optional

        # create pipeline
        pipeline = hp.RayTracingPipeline(
            self._shaders, maxRecursionDepth=maxRecursionDepth
        )
        # create sbt
        sbt = {}
        for shaderType, entries in self._sbt.items():
            sbt[shaderType] = pipeline.createShaderBindingTable(entries)
        # to prevent from creating the same pipeline twice, delete everything
        del self._shaders
        del self._sbt
        # done
        return pipeline, sbt
