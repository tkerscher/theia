import numpy as np
import hephaistos as hp
import importlib.resources
import os
import os.path
import sys
import pytest

# needed to discover common package...
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(scope="session", autouse=True)
def gpu():
    if hp.isRaytracingSupported():
        # raise RuntimeError("This system does not support ray tracing!")
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

    def compileTestShader(self, shader, preamble="", headers={}) -> bytes:
        source = preamble + "\n" + self.getTestShader(shader)
        code = self.compiler.compile(source, headers)
        return code

    def createTestProgram(self, shader, preamble="", headers={}) -> hp.Program:
        code = self.compileTestShader(shader, preamble, headers)
        return hp.Program(code)


@pytest.fixture(scope="session")
def shaderUtil():
    return ShaderUtil()


@pytest.fixture(scope="session")
def testDataDir():
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="session")
def dataDir():
    return importlib.resources.files("theia").joinpath("data")


@pytest.fixture(scope="function")
def rng():
    # create a fresh generator so each test is reproducible and independent
    return np.random.default_rng(0xC0FFEE)
