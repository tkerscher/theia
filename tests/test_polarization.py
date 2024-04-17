import numpy as np
import hephaistos as hp
from scipy.spatial.transform import Rotation as R

from ctypes import *
from hephaistos.glsl import vec3, vec4, stackVector

import theia.material

from numpy.lib.recfunctions import structured_to_unstructured


def test_polarizationRotate_phi(rng, shaderUtil):
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

    # create and run test
    program = shaderUtil.createTestProgram("polarization.rotate.phi.test.glsl")
    program.bindParams(QueryBuffer=query_tensor, ResultBuffer=result_tensor)
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


def test_polarizationRotate_dir(rng, shaderUtil):
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
    program = shaderUtil.createTestProgram("polarization.rotate.dir.test.glsl")
    program.bindParams(QueryBuffer=query_tensor, ResultBuffer=result_tensor)
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


def test_phaseMatrix(rng, shaderUtil):
    N = 32 * 1024

    # create medium for look up
    class WaterModel(
        theia.material.WaterBaseModel,
        theia.material.HenyeyGreensteinPhaseFunction,
        theia.material.KokhanovskyOceanWaterPhaseMatrix,
        theia.material.MediumModel,
    ):
        def __init__(self) -> None:
            theia.material.WaterBaseModel.__init__(self, 5.0, 1000.0, 35.0)
            theia.material.HenyeyGreensteinPhaseFunction.__init__(self, 0.6)
            theia.material.KokhanovskyOceanWaterPhaseMatrix.__init__(
                self, p90=0.66, theta0=0.25, alpha=4.0, xi=25.6  # voss measurement fit
            )

        ModelName = "water"

    water_model = WaterModel()
    water = water_model.createMedium()
    store = theia.material.MaterialStore([], media=[water])

    class Push(Structure):
        _fields_ = [("medium", c_uint64)]

    push = Push(medium=store.media["water"])

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
    program = shaderUtil.createTestProgram("polarization.phaseMatrix.test.glsl")
    program.bindParams(QueryBuffer=query_tensor, ResultBuffer=result_tensor)
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )
    results = structured_to_unstructured(result_buffer.numpy())

    # calculated expected results
    ones = np.ones(N)
    zero = np.zeros(N)
    m12 = zero if (m := water_model.phase_m12(cos_theta)) is None else m
    m22 = zero if (m := water_model.phase_m22(cos_theta)) is None else m
    m33 = zero if (m := water_model.phase_m33(cos_theta)) is None else m
    m34 = zero if (m := water_model.phase_m34(cos_theta)) is None else m
    phase_matrices = np.stack([
        ones, m12, zero, zero,
        m12, m22, zero, zero,
        zero, zero, m33, m34,
        zero, zero, -m34, m33,
    ], axis=-1).reshape((N, 4, 4))
    expected = np.einsum("ijk,ik->ij", phase_matrices, stokes)
    # check result
    assert np.abs(results - expected).max() < 1e-3  # TODO: maybe a bit large?
