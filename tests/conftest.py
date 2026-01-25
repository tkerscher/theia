import numpy as np
import hephaistos as hp
import importlib.resources
import os
import os.path
import pathlib
import pytest
import sys

import theia
import theia.compiler

# needed to discover common package...
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# add test shader to include dirs
theia.compiler.addIncludeDir(pathlib.Path(os.path.basename(__file__) + "/shader/"))


# @pytest.fixture(scope="session", autouse=True)
# def gpu():
#     if hp.isRaytracingSupported():
#         # raise RuntimeError("This system does not support ray tracing!")
#         hp.enableRaytracing()
#     # force device initialization
#     hp.getCurrentDevice()


@pytest.fixture(scope="function", autouse=True)
def cleanup_gpu():
    # first run test
    yield

    # cleanup
    hp.destroyResources()


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
