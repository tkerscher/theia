import numpy as np
import hephaistos as hp

# needed for side effects
import theia


def test_signBit(rng, shaderUtil):
    N = 32 * 256

    # allocate memory
    query_buffer = hp.FloatBuffer(N)
    query_tensor = hp.FloatTensor(N)
    result_buffer = hp.FloatBuffer(N)
    result_tensor = hp.FloatTensor(N)

    # fill queries
    floats = (rng.random(N, dtype=np.float32) - 0.5) * 1e36
    # special values: +/- 0.0
    floats[0] = 0.0
    floats[1] = -0.0
    query_buffer.numpy()[:] = floats

    # create test program
    program = shaderUtil.createTestProgram("math.signBit.test.glsl")
    program.bindParams(FloatIn=query_tensor, FloatOut=result_tensor)
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )
    results = result_buffer.numpy()

    # check result
    assert np.all(results == np.copysign(1.0, floats))
