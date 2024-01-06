import numpy as np
import hephaistos as hp

from ctypes import *
from hephaistos.glsl import vec3, uvec2, stackVector

import theia.material
import theia.random

from common.models import WaterModel

from numpy.lib.recfunctions import structured_to_unstructured
from theia.util import packUint64


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


def test_reflectance(rng, shaderUtil):
    N = 32 * 1024

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
    mat = mats["material"]

    # define types
    class Query(Structure):
        _fields_ = [("dir", c_float * 3), ("nrm", c_float * 3), ("lambda", c_float)]

    # reserve memory
    query_buffer = hp.ArrayBuffer(Query, N)
    query_tensor = hp.ArrayTensor(Query, N)
    result_buffer = hp.FloatBuffer(N)
    result_tensor = hp.FloatTensor(N)

    # fill input
    queries = query_buffer.numpy()
    phi_in = rng.random(N) * 2.0 * np.pi
    cos_theta_in = rng.random(N) * 2.0 - 1.0
    sin_theta_in = np.sqrt(1.0 - cos_theta_in**2)
    queries["dir"][:, 0] = sin_theta_in * np.cos(phi_in)
    queries["dir"][:, 1] = sin_theta_in * np.sin(phi_in)
    queries["dir"][:, 2] = cos_theta_in
    phi_nrm = rng.random(N) * 2.0 * np.pi
    cos_theta_nrm = rng.random(N) * 2.0 - 1.0
    sin_theta_nrm = np.sqrt(1.0 - cos_theta_nrm**2)
    queries["nrm"][:, 0] = sin_theta_nrm * np.cos(phi_nrm)
    queries["nrm"][:, 1] = sin_theta_nrm * np.sin(phi_nrm)
    queries["nrm"][:, 2] = cos_theta_nrm
    queries["lambda"] = 300.0 * rng.random(N) + 400.0

    # create program and run it
    program = shaderUtil.createTestProgram("scatter.reflectance.test.glsl")
    program.bindParams(QueryBuffer=query_tensor, ResultBuffer=result_tensor)
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatchPush(bytes(packUint64(mat)), N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )

    # calculate expected reflectance
    cos_theta = np.multiply(queries["dir"], queries["nrm"]).sum(-1)
    n_inside = inside.refractive_index(queries["lambda"])
    n_outside = outside.refractive_index(queries["lambda"])
    mask = cos_theta <= 0.0
    n_i = n_inside.copy()
    n_i[mask] = n_outside[mask]
    n_t = n_outside.copy()
    n_t[mask] = n_inside[mask]
    cos_i = np.abs(cos_theta)
    r = reflectance(cos_i, n_i, n_t)

    # check result
    results = result_buffer.numpy()
    assert results.min() >= 0.0
    assert results.max() <= 1.0
    # give here a little bit more error, since we also do table lookups
    assert np.abs(r - result_buffer.numpy()).max() <= 1e-3


def test_volumeScatter(rng, shaderUtil):
    N = 32 * 1024
    N_EMPTY = 32 * 64

    # create medium
    model = WaterModel()
    water = model.createMedium()
    empty = theia.material.MediumModel().createMedium(name="empty")
    tensor, _, media = theia.material.bakeMaterials(media=[water, empty])

    # define types
    class Query(Structure):
        _fields_ = [("dir", c_float * 3), ("medium", c_uint32 * 2)]

    class Result(Structure):
        _fields_ = [("dir", c_float * 3), ("prob", c_float)]

    # allocate memory
    inputBuffer = hp.ArrayBuffer(Query, N)
    inputTensor = hp.ArrayTensor(Query, N)
    outputBuffer = hp.ArrayBuffer(Result, N)
    outputTensor = hp.ArrayTensor(Result, N)
    # create rng
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    philox.update(0)

    # create program
    headers = {"rng.glsl": philox.sourceCode}
    program = shaderUtil.createTestProgram(
        "scatter.volume.scatter.test.glsl", headers=headers
    )
    # bind params
    philox.bindParams(program, 0)
    program.bindParams(Input=inputTensor, Output=outputTensor)

    # fill input
    queries = inputBuffer.numpy()
    phi = rng.random(N) * 2.0 * np.pi
    cos_theta_in = rng.random(N) * 2.0 - 1.0
    sin_theta_in = np.sqrt(1.0 - cos_theta_in**2)
    queries["dir"][:, 0] = sin_theta_in * np.cos(phi)
    queries["dir"][:, 1] = sin_theta_in * np.sin(phi)
    queries["dir"][:, 2] = cos_theta_in
    queries["medium"][N_EMPTY:] = packUint64(media["water"]).value
    queries["medium"][:N_EMPTY] = packUint64(media["empty"]).value

    # run program
    (
        hp.beginSequence()
        .And(hp.updateTensor(inputBuffer, inputTensor))
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(outputTensor, outputBuffer))
        .Submit()
        .wait()
    )

    # test result
    result = outputBuffer.numpy()
    dir_in = queries["dir"]
    dir_out = result["dir"]
    cos_theta = np.multiply(dir_in, dir_out).sum(-1)
    # test lambertian disitribution
    bins = np.histogram(cos_theta[:N_EMPTY], bins=10, range=(-1, 1))[0]
    assert np.abs(bins / N_EMPTY - 0.1).max() < 0.1  # low statistic
    assert np.allclose(result["prob"][:N_EMPTY], 1.0 / (4.0 * np.pi))
    # test henyey greenstein via E[cos_theta] = g
    assert np.abs(cos_theta[N_EMPTY:].mean() - model.g) < 5e-3
    p_exp = np.exp(model.log_phase_function(cos_theta[N_EMPTY:]))
    assert np.abs(p_exp - result["prob"][N_EMPTY:]).max() < 5e-5

    # check if scattered phi is truly random
    # create a local euclidean cosy from input direction as one axis
    # the ratio of the projection onto the other two axis is the tan
    # since we expect uniform, we are free to rotate around the input direction
    # See prbt v4: chapter 3.3.3
    s = np.copysign(1.0, dir_in[:, 2])
    a = -1.0 / (s + dir_in[:, 2])
    b = a * dir_in[:, 0] * dir_in[:, 1]
    x_ax = np.stack([1.0 + s * dir_in[:, 0] ** 2 * a, s * b, -s * dir_in[:, 0]], -1)
    y_ax = np.stack([b, s + dir_in[:, 1] ** 2 * a, -dir_in[:, 1]], -1)
    # normalize to be safe
    x_ax = x_ax / np.sqrt(np.square(x_ax).sum(-1))[:, None]
    y_ax = y_ax / np.sqrt(np.square(y_ax).sum(-1))[:, None]
    # project onto axis
    x = np.multiply(x_ax, dir_out).sum(-1)
    y = np.multiply(y_ax, dir_out).sum(-1)
    phi = np.arctan2(y, x)
    # check if phi is uniform
    hist = np.histogram(phi, bins=10, range=(-np.pi, np.pi))[0]
    # check for uniform
    assert np.abs((hist / N) - (1.0 / 10)).max() < 0.01  # low statistics


def test_volumeScatterProb(rng, shaderUtil):
    N = 32 * 256

    # create medium
    model = WaterModel()
    water = model.createMedium()
    empty = theia.material.MediumModel().createMedium(name="empty")
    tensor, _, media = theia.material.bakeMaterials(media=[water, empty])

    # define type
    class Query(Structure):
        _fields_ = [
            ("inDir", c_float * 3),
            ("outDir", c_float * 3),
            ("medium", c_uint32 * 2),
        ]

    # allocate memory
    inputBuffer = hp.ArrayBuffer(Query, N)
    inputTensor = hp.ArrayTensor(Query, N)
    outputBuffer = hp.FloatBuffer(N)
    outputTensor = hp.FloatTensor(N)

    # create program
    program = shaderUtil.createTestProgram("scatter.volume.prob.test.glsl")
    # bind params
    program.bindParams(Input=inputTensor, Output=outputTensor)

    # fill input
    queries = inputBuffer.numpy()
    phi_in = rng.random(N) * 2.0 * np.pi
    cos_theta_in = rng.random(N) * 2.0 - 1.0
    sin_theta_in = np.sqrt(1.0 - cos_theta_in**2)
    queries["inDir"][:, 0] = sin_theta_in * np.cos(phi_in)
    queries["inDir"][:, 1] = sin_theta_in * np.sin(phi_in)
    queries["inDir"][:, 2] = cos_theta_in
    queries["medium"][: (N // 2)] = packUint64(media["water"]).value
    queries["medium"][(N // 2) :] = packUint64(media["empty"]).value
    phi_out = rng.random(N) * 2.0 * np.pi
    cos_theta_out = rng.random(N) * 2.0 - 1.0
    sin_theta_out = np.sqrt(1.0 - cos_theta_out**2)
    queries["outDir"][:, 0] = sin_theta_out * np.cos(phi_out)
    queries["outDir"][:, 1] = sin_theta_out * np.sin(phi_out)
    queries["outDir"][:, 2] = cos_theta_out

    # run program
    (
        hp.beginSequence()
        .And(hp.updateTensor(inputBuffer, inputTensor))
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(outputTensor, outputBuffer))
        .Submit()
        .wait()
    )

    # calc expected prob using models
    cos_theta = np.multiply(queries["inDir"], queries["outDir"]).sum(-1)
    p = np.empty(N)
    p[: (N // 2)] = np.exp(model.log_phase_function(cos_theta[: (N // 2)]))
    p[(N // 2) :] = 1.0 / (4.0 * np.pi)  # lambertian / empty model
    # check result
    result = outputBuffer.numpy()
    assert np.abs(result - p).max() < 5e-5  # extra error from interpolation <-> model
