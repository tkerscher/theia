import numpy as np
import hephaistos as hp
from ctypes import Structure, c_float
from hephaistos.glsl import vec3, stackVector
from hephaistos.pipeline import runPipelineStage
from numpy.lib.recfunctions import structured_to_unstructured
from theia.random import PhiloxRNG, RNGBufferSink


def test_sampleSphere(rng, shaderUtil):
    N = 32 * 256

    # define types
    class Result(Structure):
        _fields_ = [("dir", vec3), ("prob", c_float)]

    class Push(Structure):
        _fields_ = [("position", vec3), ("radius", c_float)]

    # allocate memory
    inputBuffer = hp.ArrayBuffer(vec3, N)
    inputTensor = hp.ArrayTensor(vec3, N)
    outputBuffer = hp.ArrayBuffer(Result, N)
    outputTensor = hp.ArrayTensor(Result, N)
    # create rng
    philox = PhiloxRNG(key=0xC01DC0FFEE)  # need 2N samples
    generator = RNGBufferSink(philox, N // 2, 16)
    runPipelineStage(generator)

    # create program
    program = shaderUtil.createTestProgram("sphere.sample.test.glsl")
    # bind params
    program.bindParams(Input=inputTensor, Output=outputTensor, RNG=generator.tensor)

    # create push
    cx, cy, cz, r = 10.0, 5.0, 27.0, 3.5
    push = Push(position=vec3(x=cx, y=cy, z=cz), radius=r)
    # fill input
    inBuf = inputBuffer.numpy()
    phi = rng.random(N) * 2.0 * np.pi
    cos_theta = 2.0 * rng.random(N) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    rho = rng.random(N) * 50.0 + r + 1.5
    x = rho * sin_theta * np.cos(phi) + cx
    y = rho * sin_theta * np.sin(phi) + cy
    z = rho * cos_theta + cz
    inBuf[:] = stackVector([x, y, z], vec3)

    # run program
    (
        hp.beginSequence()
        .And(hp.updateTensor(inputBuffer, inputTensor))
        .Then(program.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(outputTensor, outputBuffer))
        .Submit()
        .wait()
    )

    # test results
    o = outputBuffer.numpy()
    dx, dy, dz = x - cx, y - cy, z - cz
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    sin_cone = r / dist
    cos_cone = np.sqrt(1.0 - sin_cone**2)
    prob_expected = 1.0 / (2.0 * np.pi * (1.0 - cos_cone))
    assert np.allclose(o["prob"], prob_expected, 1e-4)
    center_dir = np.stack([dx, dy, dz], -1)
    center_dir /= np.sqrt(np.square(center_dir).sum(-1))[:, None]
    sample_dir = structured_to_unstructured(o["dir"])
    assert np.allclose(np.square(sample_dir).sum(-1), 1.0)
    cos_sample = -np.multiply(center_dir, sample_dir).sum(-1)
    assert np.all(cos_sample > cos_cone)  # larger cos => smaller angle


def test_sampleProb(rng, shaderUtil):
    N = 32 * 256

    # define Types
    class Query(Structure):
        _fields_ = [("observer", vec3), ("direction", vec3)]

    class Push(Structure):
        _fields_ = [("position", vec3), ("radius", c_float)]

    # allocate memory
    inputBuffer = hp.ArrayBuffer(Query, N)
    inputTensor = hp.ArrayTensor(Query, N)
    outputBuffer = hp.FloatBuffer(N)
    outputTensor = hp.FloatTensor(N)

    # create program
    program = shaderUtil.createTestProgram("sphere.prob.test.glsl")
    # bind params
    program.bindParams(Input=inputTensor, Output=outputTensor)

    # create push
    cx, cy, cz, r = 10.0, 5.0, 27.0, 3.5
    push = Push(position=vec3(x=cx, y=cy, z=cz), radius=r)
    # fill input
    inBuf = inputBuffer.numpy()
    phi = rng.random(N) * 2.0 * np.pi
    theta = np.arccos(2.0 * rng.random(N) - 1.0)
    rho = rng.random(N) * 50.0 + r + 1.5
    x = rho * np.sin(theta) * np.cos(phi) + cx
    y = rho * np.sin(theta) * np.sin(phi) + cy
    z = rho * np.cos(theta) + cz
    inBuf["observer"] = stackVector([x, y, z], vec3)
    theta_jitter = rng.normal(scale=0.1, size=N)
    dx = -np.sin(theta + theta_jitter) * np.cos(phi)
    dy = -np.sin(theta + theta_jitter) * np.sin(phi)
    dz = -np.cos(theta + theta_jitter)
    inBuf["direction"] = stackVector([dx, dy, dz], vec3)

    # run program
    (
        hp.beginSequence()
        .And(hp.updateTensor(inputBuffer, inputTensor))
        .Then(program.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(outputTensor, outputBuffer))
        .Submit()
        .wait()
    )

    # test results
    p = outputBuffer.numpy()
    dx, dy, dz = x - cx, y - cy, z - cz
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    sin_cone = r / dist
    cos_cone = np.sqrt(1.0 - sin_cone**2)
    cos_dir = np.cos(theta_jitter)
    hit_mask = cos_dir >= cos_cone
    prob_expected = 1.0 / (2.0 * np.pi * (1.0 - cos_cone))
    assert np.allclose(p[hit_mask], prob_expected[hit_mask], 1e-4)
    # check if pos prob, actually inside view cone; error a bit large...
    assert np.max(cos_cone - cos_dir, initial=0.0, where=(p != 0.0)) < 1e-3
    # and if zero prob, outside view cone
    assert np.min(cos_cone - cos_dir, initial=1.0, where=(p == 0.0)) > 0.0


def test_sampleProbNorm(rng, shaderUtil):
    N = 32 * 32768

    # define Types
    class Query(Structure):
        _fields_ = [("observer", vec3), ("direction", vec3)]

    class Push(Structure):
        _fields_ = [("position", vec3), ("radius", c_float)]

    # allocate memory
    inputBuffer = hp.ArrayBuffer(Query, N)
    inputTensor = hp.ArrayTensor(Query, N)
    outputBuffer = hp.FloatBuffer(N)
    outputTensor = hp.FloatTensor(N)

    # create program
    program = shaderUtil.createTestProgram("sphere.prob.test.glsl")
    # bind params
    program.bindParams(Input=inputTensor, Output=outputTensor)

    # create push
    cx, cy, cz, r = 10.0, 5.0, 27.0, 3.5
    push = Push(position=vec3(x=cx, y=cy, z=cz), radius=r)
    # fill input
    inBuf = inputBuffer.numpy()
    phi = rng.random(N) * 2.0 * np.pi
    theta = np.arccos(2.0 * rng.random(N) - 1.0)
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)
    inBuf["direction"] = stackVector([dx, dy, dz], vec3)

    # run program
    (
        hp.beginSequence()
        .And(hp.updateTensor(inputBuffer, inputTensor))
        .Then(program.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(outputTensor, outputBuffer))
        .Submit()
        .wait()
    )

    # test if probability is normalized
    norm = 2.0 * np.pi * outputBuffer.numpy().sum() / N
    assert np.abs(norm - 1.0) < 1e-2  # large error since we dont use many samples


def test_intersectSphere(rng, shaderUtil):
    N = 32 * 256
    pos, r = (-0.1, 0.2, 3.0), 2.0

    # define types
    class Query(Structure):
        _fields_ = [("pos", vec3), ("dir", vec3)]

    class Push(Structure):
        _fields_ = [("pos", vec3), ("r", c_float)]

    # allocate memory
    inputBuffer = hp.ArrayBuffer(Query, N)
    inputTensor = hp.ArrayTensor(Query, N)
    resultBuffer = hp.FloatBuffer(N)
    resultTensor = hp.FloatTensor(N)
    push = Push(pos=vec3(*pos), r=r)

    # create program
    program = shaderUtil.createTestProgram("sphere.intersect.test.glsl")
    program.bindParams(Queries=inputTensor, Result=resultTensor)

    # fill input
    x = 2.0 * rng.random(N) - 1.0
    y = 2.0 * rng.random(N) - 1.0
    z = 2.0 * rng.random(N) - 1.0
    cos_theta = 1.0 - 0.5 * rng.random(N)  # limit to upper sphere
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * rng.random(N)
    queries = inputBuffer.numpy()
    queries["pos"] = stackVector([x, y, z], vec3)
    queries["dir"] = stackVector(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], vec3
    )

    # run test
    (
        hp.beginSequence()
        .And(hp.updateTensor(inputBuffer, inputTensor))
        .Then(program.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(resultTensor, resultBuffer))
        .Submit()
        .wait()
    )

    # calc expected result
    center = np.array(pos)
    pos = structured_to_unstructured(queries["pos"])
    dir = structured_to_unstructured(queries["dir"])
    f = pos - center
    discrim = np.multiply(dir, f).sum(-1) ** 2 - np.square(f).sum(-1) + r**2
    hit = discrim >= 0.0
    # check results
    t = resultBuffer.numpy()
    hit_pos = (pos + dir * t[:, None])[hit]
    assert np.all(np.isinf(t[~hit]))
    assert np.allclose(np.square(hit_pos - center).sum(-1), r**2)
