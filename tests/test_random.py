import numpy as np
import theia.random
import theia.util

from hephaistos.pipeline import RetrieveTensorStage, runPipeline


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
