import numpy as np
import hephaistos as hp


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
        .Submit().wait()
    )

    # checking randomness is a bit much for a unit test
    # let's check for uniform instead
    # NOTE: a simple linear increase would also pass this

    hist = np.histogram(buffer.numpy(), bins=20)[0]
    max_dev = np.abs((hist * 20.0 / len(buffer.numpy())) - 1.0).max()
    assert max_dev < 0.005  # TODO: What is a reasonable value here?
