import numpy as np
import hephaistos as hp
import theia.material
from ctypes import *
from hephaistos.glsl import vec3, stackVector
from numpy.lib.recfunctions import structured_to_unstructured


def test_scatterDir(rng, shaderUtil):
    N = 32 * 1024
    # reserve memory
    class Query(Structure):
        _fields_ = [("inDir", vec3), ("cos_theta", c_float), ("phi", c_float)]

    query_buffer = hp.ArrayBuffer(Query, N)
    query_tensor = hp.ArrayTensor(Query, N)
    result_buffer = hp.ArrayBuffer(vec3, N)
    result_tensor = hp.ArrayTensor(vec3, N)

    # fill queries
    theta = rng.random(N) * np.pi
    phi = rng.random(N) * 2.0 * np.pi
    queries = query_buffer.numpy()
    queries["inDir"] = stackVector(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), vec3
    )
    queries["cos_theta"] = 2.0 * rng.random(N) - 1.0
    queries["phi"] = 2.0 * np.pi * rng.random(N)

    # create and run test
    program = shaderUtil.createTestProgram("scatter.scatterDir.test.glsl")
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

    # check cos_theta
    inDir = structured_to_unstructured(queries["inDir"])
    dots = np.multiply(inDir, results).sum(-1)
    assert np.abs(dots - queries["cos_theta"]).max() < 5e-7

    # check phi

    # create transformation inDir -> e_z
    # this allows to check the phi distribution
    m11 = np.cos(theta) * np.cos(phi)
    m12 = np.cos(theta) * np.sin(phi)
    m13 = -np.sin(theta)
    m21 = -np.sin(phi)
    m22 = np.cos(phi)
    m23 = np.zeros_like(theta)
    m31 = np.sin(theta) * np.cos(phi)
    m32 = np.sin(theta) * np.sin(phi)
    m33 = np.cos(theta)
    trafo = np.stack(
        [
            np.stack([m11, m21, m31], axis=-1),
            np.stack([m12, m22, m32], axis=-1),
            np.stack([m13, m23, m33], axis=-1),
        ],
        axis=-1,
    )

    dir_out_proj = np.einsum("ikj,ij->ik", trafo, results)
    phi_out = np.arctan2(dir_out_proj[:, 1], dir_out_proj[:, 0])
    # we can't compare the input phi with the calculated one
    # as we allow phi=0 to point in any (legal) direction
    # instead we check the distribution to be uniform
    bins = 32
    counts, _ = np.histogram(phi_out, bins=bins)
    counts = counts / N
    thres = 1.0 / np.sqrt(N / bins)  # exptected error
    assert np.abs(counts - 1.0 / bins).max() < thres


def reflectance(cos_i, n_i, n_t):
    """implements fresnel equations"""
    sin_i = np.sqrt(np.maximum(1.0 - cos_i * cos_i, np.zeros_like(cos_i)))
    sin_t = sin_i * n_i / n_t
    cos_t = np.sqrt(np.maximum(1.0 - sin_t * sin_t, np.zeros_like(sin_t)))

    rs = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t)
    rp = (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t)
    return 0.5 * (rs * rs + rp * rp)


class Query(Structure):
    _fields_ = [("direction", vec3), ("normal", vec3), ("wavelength", c_float)]


class Scene(Structure):
    _fields_ = [("mat", c_uint64)]


def prepareRayQueries(N, rng):
    # we only need the refractive index so we can skip the phase functions
    class WaterModel(theia.material.WaterBaseModel, theia.material.MediumModel):
        def __init__(self) -> None:
            theia.material.WaterBaseModel.__init__(self, 5.0, 1000.0, 35.0)

    inside = theia.material.BK7Model()
    outside = WaterModel()
    material = theia.material.Material(
        "material",
        inside.createMedium(name="glass"),
        outside.createMedium(name="water"),
    )
    # bake material
    mat_tensor, mats, media = theia.material.bakeMaterials(material)
    scene = Scene(mat=mats["material"])

    # reserve memory
    class Query(Structure):
        _fields_ = [("direction", vec3), ("normal", vec3), ("wavelength", c_float)]

    query_buffer = hp.ArrayBuffer(Query, N)
    # fill query buffer
    queries = query_buffer.numpy()
    d_theta = rng.random(N, np.float32) * np.pi
    d_phi = rng.random(N, np.float32) * 2.0 * np.pi
    n_theta = rng.random(N, np.float32) * np.pi
    n_phi = rng.random(N, np.float32) * 2.0 * np.pi
    queries["direction"] = stackVector(
        (
            np.sin(d_theta) * np.cos(d_phi),
            np.sin(d_theta) * np.sin(d_phi),
            np.cos(d_theta),
        ),
        vec3,
    )
    queries["normal"] = stackVector(
        (
            np.sin(n_theta) * np.cos(n_phi),
            np.sin(n_theta) * np.sin(n_phi),
            np.cos(n_theta),
        ),
        vec3,
    )
    queries["wavelength"] = rng.random(N, np.float32) * 600.0 + 200.0

    # calculate expected reflectance
    directions = structured_to_unstructured(queries["direction"])
    normals = structured_to_unstructured(queries["normal"])
    cos_theta = np.multiply(directions, normals).sum(-1)
    n_inside = inside.refractive_index(queries["wavelength"])
    n_outside = outside.refractive_index(queries["wavelength"])
    mask = cos_theta <= 0.0
    n_i = n_inside.copy()
    n_i[mask] = n_outside[mask]
    n_t = n_outside.copy()
    n_t[mask] = n_inside[mask]
    cos_i = np.abs(cos_theta)
    r = reflectance(cos_i, n_i, n_t)

    return query_buffer, scene, n_i, n_t, r, mat_tensor


def test_reflectance(rng, shaderUtil):
    N = 32 * 1024
    query_buffer, scene, n_i, n_t, r, mat_tensor = prepareRayQueries(N, rng)

    # reserve memory
    query_tensor = hp.ArrayTensor(Query, N)
    result_buffer = hp.FloatBuffer(N)
    result_tensor = hp.FloatTensor(N)

    # create program and run it
    program = shaderUtil.createTestProgram("scatter.reflectance.test.glsl")
    program.bindParams(QueryBuffer=query_tensor, ResultBuffer=result_tensor)
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatchPush(bytes(scene), N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )

    # check result
    results = result_buffer.numpy()
    assert results.min() >= 0.0
    assert results.max() <= 1.0
    # give here a little bit more error, since we also do table lookups
    assert np.abs(r - result_buffer.numpy()).max() <= 1e-3


def test_reflectMaterial(rng, shaderUtil):
    N = 32 * 1024
    query_buffer, scene, n_i, n_t, r, mat_tensor = prepareRayQueries(N, rng)

    # reserve memory
    class Result(Structure):
        _fields_ = [("direction", vec3), ("transmission", c_float)]

    query_tensor = hp.ArrayTensor(Query, N)
    result_buffer = hp.ArrayBuffer(Result, N)
    result_tensor = hp.ArrayTensor(Result, N)

    # create program and run it
    program = shaderUtil.createTestProgram("scatter.reflectMaterial.test.glsl")
    program.bindParams(QueryBuffer=query_tensor, ResultBuffer=result_tensor)
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatchPush(bytes(scene), N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )

    # recreate expected result
    queries = query_buffer.numpy()
    results = result_buffer.numpy()
    directions = structured_to_unstructured(queries["direction"])
    normals = structured_to_unstructured(queries["normal"])
    # check result
    assert results["transmission"].min() >= 0.0
    assert results["transmission"].max() <= 1.0
    # give here a little bit more error, since we also do table lookups
    assert np.abs(r - results["transmission"]).max() <= 1e-3

    # reflected direction
    dots = np.multiply(directions, normals).sum(-1)
    dir_exp = directions - 2.0 * dots[:, None] * normals
    dir_out = structured_to_unstructured(results["direction"])
    # check result
    assert np.abs(dir_exp - dir_out).max() <= 5e-7


def test_transmitMaterial(rng, shaderUtil):
    N = 32 * 1024
    query_buffer, scene, n_i, n_t, r, mat_tensor = prepareRayQueries(N, rng)

    # reserve memory
    class Result(Structure):
        _fields_ = [
            ("direction", vec3),
            ("transmission", c_float),
            ("refractive_index", c_float),
            ("match_consts", c_float),
        ]

    query_tensor = hp.ArrayTensor(Query, N)
    result_buffer = hp.ArrayBuffer(Result, N)
    result_tensor = hp.ArrayTensor(Result, N)

    # create program and run it
    program = shaderUtil.createTestProgram("scatter.transmitMaterial.test.glsl")
    program.bindParams(QueryBuffer=query_tensor, ResultBuffer=result_tensor)
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatchPush(bytes(scene), N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )

    # recreate expected result
    queries = query_buffer.numpy()
    results = result_buffer.numpy()
    directions = structured_to_unstructured(queries["direction"])
    normals = structured_to_unstructured(queries["normal"])
    # since we transmit, it's actually 1 - reflectance
    r = 1.0 - r
    # check result
    assert results["transmission"].min() >= 0.0
    assert results["transmission"].max() <= 1.0
    # give here a little bit more error, since we also do table lookups
    assert np.abs(r - results["transmission"]).max() <= 1e-3

    # transmitted direction
    dots = np.multiply(directions, normals).sum(-1)
    # normal and direction must point in opposite direction -> flip if necessary
    normals *= -np.sign(dots)[:, None]
    dots = np.multiply(directions, normals).sum(-1)
    eta = n_i / n_t
    k = 1.0 - eta * eta * (1.0 - dots * dots)
    mask = k >= 0.0
    dir_exp = np.zeros((mask.sum(),))
    eta = eta[mask]
    dots = dots[mask]
    k = k[mask]
    normals = normals[mask]
    directions = directions[mask]
    dir_exp = eta[:, None] * directions - (eta * dots + np.sqrt(k))[:, None] * normals
    dir_out = structured_to_unstructured(results["direction"])
    # check result (give a bit more error since we also use lookups)
    assert np.abs(dir_exp - dir_out[mask]).max() <= 1e-4
    # illegal transmission (total reflection) are mapped to NaN
    assert np.all(np.isnan(dir_out[~mask]))

    # check if we end up correct material
    assert np.abs(n_t - results["refractive_index"]).max() <= 1e-5

    # check if the material lookup is correct (happened in shader)
    assert np.all(results["match_consts"] >= 0.0)