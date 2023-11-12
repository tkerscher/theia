import numpy as np
import theia.random
import theia.scheduler
import theia.util

from ctypes import c_float
from numpy.ctypeslib import as_array


def test_philox():
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    generator = theia.random.RNGBufferSink(philox, 4096, 8192)
    retrieve = theia.scheduler.RetrieveTensorStage(generator.tensor)

    theia.scheduler.runPipeline([philox, generator, retrieve])

    data = (c_float * generator.capacity).from_address(retrieve.buffer().address)
    array = as_array(data)

    # checking randomness is a bit much for a unit test
    # let's check for uniform instead
    # NOTE: a simple linear increase would also pass this

    hist = np.histogram(array, bins=20)[0]
    max_dev = np.abs((hist * 20.0 / generator.capacity) - 1.0).max()
    assert max_dev < 0.005  # TODO: What is a reasonable value here?


# helper test to make debugging easier
def test_uvec4_conversion():
    i = 0xC0110FFC0FFEE
    assert theia.util.uvec4ToInt(theia.util.intToUvec4(i)) == i
