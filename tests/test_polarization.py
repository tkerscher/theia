import numpy as np
import hephaistos as hp

from ctypes import *
from hephaistos.glsl import vec4, stackVector

import theia.material

from numpy.lib.recfunctions import structured_to_unstructured
from theia.util import packUint64


def test_polarizationRotate(rng, shaderUtil):
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
    stokes = rng.random((N,4))
    queries = query_buffer.numpy()
    queries["phi"] = phi
    queries["stokes"] = stackVector([stokes], vec4)

    # create and run test
    program = shaderUtil.createTestProgram("polarization.rotate.test.glsl")
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
    cos_phi = np.cos(2*phi)
    sin_phi = np.sin(2*phi)
    expected[:,1] = cos_phi * stokes[:,1] - sin_phi * stokes[:,2]
    expected[:,2] = sin_phi * stokes[:,1] + cos_phi * stokes[:,2]
    # check result
    assert np.abs(results - expected).max() < 5e-6


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
            theia.material.KokhanovskyOceanWaterPhaseMatrix.__init__(self,
                p90=0.66, theta0=0.25, alpha=4.0, xi=25.6 # voss measurement fit
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
    stokes = rng.random((N,4))
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
        zero, zero, -m34, m33
    ], axis=-1).reshape((N,4,4))
    expected = np.einsum("ijk,ik->ij", phase_matrices, stokes)
    # check result
    assert np.abs(results - expected).max() < 1e-3 # TODO: maybe a bit large?
