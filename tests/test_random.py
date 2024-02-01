import numpy as np
import theia.random
import theia.util

from hephaistos.pipeline import RetrieveTensorStage, runPipeline


shader = """\
float random(uint stream, uint i) {
    return 10000.0 * stream + i;
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
    ret = RetrieveTensorStage(sink.tensor)
    runPipeline([gen, sink, ret])

    samples = ret.view(0)
    streams = (np.arange(250) + 12) * 10000.0
    counts = np.arange(800) + 316
    expected = streams[:, None] + counts[None:,]

    assert np.all(samples == expected.flatten())


def test_philox():
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    generator = theia.random.RNGBufferSink(philox, 4096, 8192)
    retrieve = RetrieveTensorStage(generator.tensor)

    runPipeline([philox, generator, retrieve])

    # checking randomness is a bit much for a unit test
    # let's check for uniform instead
    # NOTE: a simple linear increase would also pass this

    data = retrieve.view(0)
    hist = np.histogram(data, bins=20)[0]
    max_dev = np.abs((hist * 20.0 / generator.capacity) - 1.0).max()
    assert max_dev < 0.005  # TODO: What is a reasonable value here?


# helper test to make debugging easier
def test_uvec4_conversion():
    i = 0xC0110FFC0FFEE
    assert theia.util.uvec4ToInt(theia.util.intToUvec4(i)) == i


def test_sobol():
    sobol = theia.random.SobolQRNG(seed=0xC0FFEE)
    # sobol is limited to 1024 samples
    generator = theia.random.RNGBufferSink(sobol, 32_768, 1024)
    retrieve = RetrieveTensorStage(generator.tensor)

    runPipeline([sobol, generator, retrieve])

    # checking randomness is a bit much for a unit test
    # let's check for uniform instead
    # NOTE: a simple linear increase would also pass this

    data = retrieve.view(0)
    hist = np.histogram(data, bins=20)[0]
    max_dev = np.abs((hist * 20.0 / generator.capacity) - 1.0).max()
    assert max_dev < 0.001  # TODO: What is a reasonable value here?
