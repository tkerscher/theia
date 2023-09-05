import hephaistos as hp
import importlib.resources
import os.path
import pytest


@pytest.fixture(scope="session", autouse=True)
def gpu():
    if not hp.isRaytracingSupported():
        raise RuntimeError("This system does not support ray tracing!")
    hp.enableRaytracing()
    # force device initialization
    hp.getCurrentDevice()


class ShaderUtil:
    """Collections of helper functions"""

    def __init__(self) -> None:
        shader_dir = str(importlib.resources.files("theia").joinpath("shader"))
        compiler = hp.Compiler()
        compiler.addIncludeDir(shader_dir)
        self.compiler = compiler

    def getTestShader(self, shader) -> str:
        path = os.path.join(os.path.dirname(__file__), "shader")
        path = os.path.join(path, shader)
        with open(path, "r") as file:
            return file.read()

    def compileTestShader(self, shader) -> bytes:
        source = self.getTestShader(shader)
        code = self.compiler.compile(source)
        return code

    def createTestProgram(self, shader) -> hp.Program:
        code = self.compileTestShader(shader)
        return hp.Program(code)


@pytest.fixture(scope="session")
def shaderUtil():
    return ShaderUtil()
