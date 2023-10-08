import numpy as np
import hephaistos as hp
import theia.random
import theia.util


def test_philox(shaderUtil):
    # Config; Should match the one in the test shader
    StreamSize = 10000
    Streams = 4096
    # allocate memory
    buffer = hp.FloatBuffer(Streams * StreamSize)
    tensor = hp.FloatTensor(Streams * StreamSize)

    # create and run program
    program = shaderUtil.createTestProgram("philox.test.glsl")
    program.bindParams(outBuf=tensor)
    (
        hp.beginSequence()
        .And(program.dispatch(Streams // 32))
        .And(hp.retrieveTensor(tensor, buffer))
        .Submit()
        .wait()
    )

    # checking randomness is a bit much for a unit test
    # let's check for uniform instead
    # NOTE: a simple linear increase would also pass this

    hist = np.histogram(buffer.numpy(), bins=20)[0]
    max_dev = np.abs((hist * 20.0 / len(buffer.numpy())) - 1.0).max()
    assert max_dev < 0.005  # TODO: What is a reasonable value here?


# helper test to make debugging easier
def test_uvec4_conversion():
    i = 0xC0110FFC0FFEE
    assert theia.util.uvec4ToInt(theia.util.intToUvec4(i)) == i


def test_philox_buffer():
    philox = theia.random.PhiloxRNG(32 * 256, 4 * 64, key=0xC0110FFC0FFEE)
    buffer1 = hp.FloatBuffer(philox.tensor.size)
    buffer2 = hp.FloatBuffer(philox.tensor.size)

    hp.beginSequence().And(philox.dispatchNext()).Then(
        hp.retrieveTensor(philox.tensor, buffer1)
    ).Then(philox.dispatchNext()).Then(
        hp.retrieveTensor(philox.tensor, buffer2)
    ).Submit().wait()

    # check wether dispatchNext() actually produced new values
    assert not np.all(buffer1.numpy() == buffer2.numpy())

    # checking randomness is a bit much for a unit test
    # let's check for uniform instead
    # NOTE: a simple linear increase would also pass this

    hist = np.histogram(buffer2.numpy(), bins=20)[0]
    max_dev = np.abs((hist * 20.0 / len(buffer2.numpy())) - 1.0).max()
    assert max_dev < 0.006  # TODO: What is a reasonable value here?
