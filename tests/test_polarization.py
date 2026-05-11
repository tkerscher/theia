import numpy as np
import hephaistos as hp
from scipy.spatial.transform import Rotation as R

from ctypes import *
from hephaistos.glsl import vec3, vec4, stackVector

import theia.material
import theia.random
from theia.compiler import createPreamble, compileShader

from numpy.lib.recfunctions import structured_to_unstructured


def test_polarizationRotate_phi(rng):
    N = 32 * 1024

    # reserve memory
    class Query(Structure):
        _fields_ = [("stokes", vec4), ("phi", c_float)]

    query_buffer = hp.ArrayBuffer(Query, N)
    query_tensor = hp.ArrayTensor(Query, N)
    result_buffer = hp.ArrayBuffer(vec4, N)
    result_tensor = hp.ArrayTensor(vec4, N)

    # fill queries
    phi = rng.random(N) * 2.0 * np.pi
    stokes = rng.random((N, 4))
    queries = query_buffer.numpy()
    queries["phi"] = phi
    queries["stokes"] = stackVector([stokes], vec4)

    store = theia.material.MaterialStore([])
    # create and run test
    code = compileShader(
        "polarization.rotate.phi.test.glsl",
        headers=store.header,
    )
    program = hp.Program(code)
    program.bindParams(QueryBuffer=query_tensor, ResultBuffer=result_tensor)
    store.bindParams(program)
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )
    results = structured_to_unstructured(result_buffer.numpy())

    # calculate expected results
    # def rot(phi):
    #     return np.array([
    #         [1.0, 0.0, 0.0, 0.0],
    #         [0.0, np.cos(2*phi), -np.sin(2*phi), 0.0],
    #         [0.0, np.sin(2*phi), np.cos(2*phi), 0.0],
    #         [0.0, 0.0, 0.0, 1.0]
    #     ])
    # expected = np.array([rot(phi) @ s for phi, s in zip(phi, stokes)])
    expected = stokes.copy()
    cos_phi = np.cos(2 * phi)
    sin_phi = np.sin(2 * phi)
    expected[:, 1] = cos_phi * stokes[:, 1] - sin_phi * stokes[:, 2]
    expected[:, 2] = sin_phi * stokes[:, 1] + cos_phi * stokes[:, 2]
    # check result
    assert np.abs(results - expected).max() < 5e-6


def test_alignPolRef():
    N = 32 * 1024

    # allocate memory
    result_buffer = hp.FloatBuffer(N * 16)
    result_tensor = hp.FloatTensor(N * 16)
    # create and prepare test program
    store = theia.material.MaterialStore([])
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    code = compileShader(
        "polarization.alignPolRef.test.glsl",
        headers={
            "random.glsl": rng.sourceCode,
            **store.header,
        },
    )
    program = hp.Program(code)
    program.bindParams(ResultBuffer=result_tensor)
    store.bindParams(program)
    rng.bindParams(program, 0)
    rng.update(0)
    # run test program
    (
        hp.beginSequence()
        .And(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )

    # check result
    result = result_buffer.numpy().reshape((N, 4, 4))
    assert np.allclose(result, np.identity(4)[None, ...], atol=5e-6)


def test_polarizationRotate_dir(rng):
    N = 32 * 1024

    # reserve memory
    class Query(Structure):
        _fields_ = [
            ("stokes", vec4),
            ("dir", vec3),
            ("ref", vec3),
            ("phi", c_float),
            ("theta", c_float),
        ]

    class Result(Structure):
        _fields_ = [("stokes", vec4), ("ref", vec3)]

    query_buffer = hp.ArrayBuffer(Query, N)
    query_tensor = hp.ArrayTensor(Query, N)
    result_buffer = hp.ArrayBuffer(Result, N)
    result_tensor = hp.ArrayTensor(Result, N)

    # fill queries
    phi = rng.random(N) * 2.0 * np.pi
    theta = rng.random(N) * np.pi
    stokes = rng.random((N, 4))
    dir = rng.random((N, 3))
    dir /= np.sqrt(np.square(dir).sum(-1))[:, None]
    # create ref perpendicular to dir
    ref = rng.random((N, 3))
    ref = np.cross(dir, ref)
    ref /= np.sqrt(np.square(ref).sum(-1))[:, None]
    queries = query_buffer.numpy()
    queries["stokes"] = stackVector([stokes], vec4)
    queries["dir"] = stackVector([dir], vec3)
    queries["ref"] = stackVector([ref], vec3)
    queries["phi"] = phi
    queries["theta"] = theta

    # create and run test
    store = theia.material.MaterialStore([])
    code = compileShader(
        "polarization.rotate.dir.test.glsl",
        headers=store.header,
    )
    program = hp.Program(code)
    program.bindParams(QueryBuffer=query_tensor, ResultBuffer=result_tensor)
    store.bindParams(program)
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )
    results = result_buffer.numpy()
    stokes_out = structured_to_unstructured(results["stokes"])
    ref_out = structured_to_unstructured(results["ref"])

    # calculate expected result
    # one caveat: scatter dir does not guarantee to exactly rotate by phi, but
    # may add a constant offset (per dir)
    cos_phi = (ref * ref_out).sum(-1)
    sin_phi = (np.cross(ref, ref_out) * dir).sum(-1)
    phi = np.arctan2(sin_phi, cos_phi)
    c2 = 2.0 * cos_phi * cos_phi - 1.0
    s2 = 2.0 * cos_phi * sin_phi
    stokes_exp = stokes.copy()
    stokes_exp[:, 1] = c2 * stokes[:, 1] - s2 * stokes[:, 2]
    stokes_exp[:, 2] = s2 * stokes[:, 1] + c2 * stokes[:, 2]
    ref_exp = R.from_rotvec(dir * phi[:, None]).apply(ref)

    # check result
    mask = np.sin(theta) > 1e-5  # degenerate cases
    assert np.abs(ref_out - ref_exp).max() < 1e-6
    assert np.abs(stokes_out - stokes_exp).max() < 1e-6
    assert np.abs(np.square(ref_out).sum(-1).max() - 1.0) < 1e-6
    assert np.abs((ref_out * dir).sum(-1)).max() < 1e-6
    # assert np.abs((ref_out * ref).sum(-1) - np.cos(phi))[mask].max() < 1e-5


def test_phaseMatrix(rng):
    N = 32 * 1024

    water_model = theia.model.PureWaterModel()
    water = water_model.createMedium()
    store = theia.material.MaterialStore([], media=[water])

    # reserve memory
    class Query(Structure):
        _fields_ = [("stokes", vec4), ("cos_theta", c_float)]

    query_buffer = hp.ArrayBuffer(Query, N)
    query_tensor = hp.ArrayTensor(Query, N)
    result_buffer = hp.ArrayBuffer(vec4, N)
    result_tensor = hp.ArrayTensor(vec4, N)

    # fill queries
    cos_theta = rng.random(N) * 2.0 - 1.0
    stokes = rng.random((N, 4))
    queries = query_buffer.numpy()
    queries["cos_theta"] = cos_theta
    queries["stokes"] = stackVector([stokes], vec4)

    # create and run test
    code = compileShader("polarization.phaseMatrix.test.glsl", headers=store.header)
    program = hp.Program(code)
    program.bindParams(QueryBuffer=query_tensor, ResultBuffer=result_tensor)
    store.bindParams(program)
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )
    results = structured_to_unstructured(result_buffer.numpy())

    # calculated expected results
    ones = np.ones(N)
    zero = np.zeros(N)
    m12 = water_model.phase_m12(cos_theta)
    m22 = water_model.phase_m22(cos_theta)
    m33 = water_model.phase_m33(cos_theta)
    m34 = zero = np.zeros(N)
    phase_matrices = np.stack(
        # fmt: off
        [
            ones, m12,  zero, zero,
            m12,  m22,  zero, zero,
            zero, zero, m33,  m34,
            zero, zero, -m34, m33,
        ],
        # fmt: on
        axis=-1,
    ).reshape((N, 4, 4))
    expected = np.einsum("ijk,ik->ij", phase_matrices, stokes)
    # check result
    assert np.abs(results - expected).max() < 1e-3  # TODO: maybe a bit large?
