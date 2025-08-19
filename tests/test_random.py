import pytest

import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl
import theia.random
import theia.util

from ctypes import Structure, c_float
from scipy.stats import kstest


shader = """\
float random_s(uint stream, uint i) {
    return 10000.0 * stream + i;
}

vec2 random2D_s(uint stream, uint dim) {
    float v = 1e4 * stream + dim;
    return vec2(v, -v);
}
"""


class DebugRNG(theia.random.RNG):
    """Debug Shader for testing the RNGBufferSink"""

    def __init__(self):
        super().__init__()

    @property
    def sourceCode(self) -> str:
        return shader


def test_rngSink():
    gen = DebugRNG()
    sink = theia.random.RNGBufferSink(gen, 250, 800, baseStream=12, baseCount=316)
    ret = pl.RetrieveTensorStage(sink.tensor)
    pl.runPipeline([gen, sink, ret])

    samples = ret.view(0).reshape(sink.shape).squeeze()
    streams = (np.arange(250) + 12) * 10000.0
    counts = np.arange(800) + 316
    expected = streams[:, None] + counts[None:,]

    assert np.all(samples == expected)


def test_rngSink_2D():
    gen = DebugRNG()
    sink = theia.random.RNGBufferSink(
        gen, 250, 800, baseStream=12, baseCount=316, sampleDim=2
    )
    retr = pl.RetrieveTensorStage(sink.tensor)
    pl.runPipeline([gen, sink, retr])

    samples = retr.view(0).reshape(sink.shape)
    streams = (np.arange(250) + 12) * 10000.0
    counts = np.arange(800) + 316
    expected = streams[:, None] + counts[None:,]
    expected = np.stack((expected, -expected), -1)

    assert np.all(samples == expected)


def test_philox():
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    generator = theia.random.RNGBufferSink(philox, 4096, 8192)
    retrieve = pl.RetrieveTensorStage(generator.tensor)

    pl.runPipeline([philox, generator, retrieve])

    # checking randomness is a bit much for a unit test
    # let's check for uniform instead
    # NOTE: a simple linear increase would also pass this

    data = retrieve.view(0)
    hist = np.histogram(data, bins=20)[0]
    max_dev = np.abs((hist * 20.0 / generator.capacity) - 1.0).max()
    assert max_dev < 0.005  # TODO: What is a reasonable value here?

    # test auto advance
    offset = philox.getParam("offset")
    assert offset == 0
    advance = 128
    philox.setParam("autoAdvance", advance)
    philox.update(0)
    assert philox.getParam("offset") - offset == advance


@pytest.mark.parametrize("alpha,gamma", [(1.0, 1.0), (0.5, 2.2), (4.8, 0.6)])
def test_gamma(alpha: float, gamma: float, shaderUtil):
    R = 32 * 1024
    N = 32 * R
    tensor = hp.FloatTensor(N)
    buffer = hp.FloatBuffer(N)

    philox = theia.random.PhiloxRNG(key=0xABBAABBA)

    class Push(Structure):
        _fields_ = [("alpha", c_float), ("gamma", c_float)]

    headers = {"random.glsl": philox.sourceCode}
    program = shaderUtil.createTestProgram("random.gamma.test.glsl", headers=headers)
    program.bindParams(Result=tensor)
    philox.bindParams(program, 0)
    philox.update(0)

    push = Push(alpha, gamma)
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), R))
        .Then(hp.retrieveTensor(tensor, buffer))
        .Submit()
        .wait()
    )

    # check result
    test = kstest(buffer.numpy(), "gamma", args=(alpha, 0.0, 1.0 / gamma))
    assert test.pvalue < 0.05


# helper test to make debugging easier
def test_uvec4_conversion():
    i = 0xC0110FFC0FFEE
    assert theia.util.uvec4ToInt(theia.util.intToUvec4(i)) == i


def test_sobol():
    sobol = theia.random.SobolQRNG(seed=0xC0FFEE)
    generator = theia.random.RNGBufferSink(sobol, 256 * 1024, 128)
    retrieve = pl.RetrieveTensorStage(generator.tensor)

    pl.runPipeline([sobol, generator, retrieve])

    # checking randomness is a bit much for a unit test
    # let's check for uniform instead
    # NOTE: a simple linear increase would also pass this

    data = retrieve.view(0)
    hist = np.histogram(data, bins=20)[0]
    max_dev = np.abs((hist * 20.0 / generator.capacity) - 1.0).max()
    assert max_dev < 1e-5  # TODO: What is a reasonable value here?

    # test auto advance
    offset = sobol.getParam("offset")
    assert offset == 0
    oldSeed = sobol.seed
    sobol.setParam("advanceSeed", 0xABBAABBA)
    sobol.update(0)
    assert oldSeed != sobol.seed
