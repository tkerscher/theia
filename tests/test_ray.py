import pytest

import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl

from ctypes import *
from hephaistos.glsl import mat4, vec3, vec4
from numpy.lib.recfunctions import structured_to_unstructured

import theia.camera
import theia.light
import theia.material
import theia.random
import theia.units as u

from theia.util import createPreamble
from common.models import WaterModel


def reflectance(cos_i, n_i, n_t):
    """implements fresnel equations"""
    sin_i = np.sqrt(np.maximum(1.0 - cos_i * cos_i, np.zeros_like(cos_i)))
    sin_t = sin_i * n_i / n_t
    cos_t = np.sqrt(np.maximum(1.0 - sin_t * sin_t, np.zeros_like(sin_t)))

    rs = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t)
    rp = (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t)
    return rs, rp


@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("polarized", [True, False])
def test_surface(shaderUtil, forward: bool, polarized: bool):
    N = 32 * 1024
    lam_min, lam_max = 200.0 * u.nm, 800.0 * u.nm
    x_nrm, y_nrm, z_nrm = 0.8, 0.36, 0.48
    v_nrm = vec3(x=x_nrm, y=y_nrm, z=z_nrm)
    normal = np.array([x_nrm, y_nrm, z_nrm])

    # reserve memory
    class Result(Structure):
        _fields_ = [
            ("lambda", c_float),
            ("orig_dir", vec3),
            ("contrib", c_float),
            ("refl_pos", vec3),
            ("refl_dir", vec3),
            ("refl_contrib", c_float),
            ("trans_pos", vec3),
            ("trans_dir", vec3),
            ("trans_contrib", c_float),
        ]

    result_buffer = hp.ArrayBuffer(Result, N)
    result_tensor = hp.ArrayTensor(Result, N)
    fetchList = [hp.retrieveTensor(result_tensor, result_buffer)]

    class PolRefResult(Structure):
        _fields_ = [
            ("refl", vec3),
            ("trans", vec3),
        ]

    polRef_buffer, polRef_tensor = None, None
    if polarized:
        polRef_buffer = hp.ArrayBuffer(PolRefResult, N)
        polRef_tensor = hp.ArrayTensor(PolRefResult, N)
        fetchList.append(hp.retrieveTensor(polRef_tensor, polRef_buffer))
    stokes_buffer, stokes_tensor = None, None

    class StokesResult(Structure):
        _fields_ = [
            ("refl", vec4),
            ("trans", vec4),
        ]

    if polarized and forward:
        stokes_buffer = hp.ArrayBuffer(StokesResult, N)
        stokes_tensor = hp.ArrayTensor(StokesResult, N)
        fetchList.append(hp.retrieveTensor(stokes_tensor, stokes_buffer))

    class MuellerResult(Structure):
        _fields_ = [
            ("refl", mat4),
            ("trans", mat4),
        ]

    mueller_buffer, mueller_tensor = None, None
    if polarized and not forward:
        mueller_buffer = hp.ArrayBuffer(MuellerResult, N)
        mueller_tensor = hp.ArrayTensor(MuellerResult, N)
        fetchList.append(hp.retrieveTensor(mueller_tensor, mueller_buffer))

    # create material
    waterModel = WaterModel()
    glassModel = theia.material.BK7Model()
    water = waterModel.createMedium(lam_min, lam_max)
    glass = glassModel.createMedium(lam_min, lam_max)
    mat = theia.material.Material("mat", glass, water)
    matStore = theia.material.MaterialStore([mat])

    # create push constants
    class Push(Structure):
        _fields_ = [
            ("material", c_uint64),
            ("normal", vec3),
            ("lam_min", c_float),
            ("lam_max", c_float),
        ]

    push = Push(
        material=matStore.material["mat"],
        normal=v_nrm,
        lam_min=lam_min,
        lam_max=lam_max,
    )

    # prepare dependencies
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    headers = {"rng.glsl": rng.sourceCode}
    pipeline = [rng]
    if forward:
        light = theia.light.SphericalLightSource(timeRange=(0.0, 0.0))
        headers["source.glsl"] = light.sourceCode
        pipeline.extend([light])
    else:
        camera = theia.camera.PointCamera()
        headers["camera.glsl"] = camera.sourceCode
        pipeline.extend([camera])
    # create test program
    preamble = createPreamble(FORWARD=forward, POLARIZATION=polarized)
    program = shaderUtil.createTestProgram("ray.surface.test.glsl", preamble, headers)
    program.bindParams(
        ResultBuffer=result_tensor,
        PolRefBuffer=polRef_tensor,
        StokesBuffer=stokes_tensor,
        MuellerBuffer=mueller_tensor,
    )
    for stage in pipeline:
        stage.update(0)
        stage._bindParams(program, 0)
    # run test
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), N // 32))
        .NextStep()
        .AndList(fetchList)
        .Submit()
        .wait()
    )
    results = result_buffer.numpy()
    # unpack results
    lam = results["lambda"]
    rayDir = structured_to_unstructured(results["orig_dir"])
    contrib = results["contrib"]
    refl_pos = structured_to_unstructured(results["refl_pos"])
    refl_dir = structured_to_unstructured(results["refl_dir"])
    refl_contrib = results["refl_contrib"]
    trans_pos = structured_to_unstructured(results["trans_pos"])
    trans_dir = structured_to_unstructured(results["trans_dir"])
    trans_contrib = results["trans_contrib"]

    # calculated expected results

    # fetch optical properties
    n_water = waterModel.refractive_index(lam)
    n_glass = glassModel.refractive_index(lam)
    cos_i = np.multiply(rayDir, normal[None, :]).sum(-1)
    inward = cos_i < 0.0
    # check rays pushed onto the correct side
    # cosines of ray direction and and position have the same sign when on
    # other side -> multiply and check for sign
    cos_re = np.multiply(refl_pos, normal[None, :]).sum(-1)
    cos_tr = np.multiply(trans_pos, normal[None, :]).sum(-1)
    assert np.all(cos_re * cos_i < 0.0)
    assert np.all(cos_tr * cos_i > 0.0)
    # calculate reflectance
    n_in = np.where(inward, n_water, n_glass)
    n_tr = np.where(inward, n_glass, n_water)
    eta = n_in / n_tr
    rs, rp = reflectance(np.abs(cos_i), n_in, n_tr)
    ts, tp = rs + 1.0, (rp + 1.0) * eta
    r = 0.5 * (rs**2 + rp**2)
    t = 1.0 - r
    if forward:
        # extra factor eta^-2
        t *= (n_tr / n_in) ** 2
    # for the math to work out we need to flip the normal to be anti to the ray
    rayNormal = -np.sign(cos_i)[:, None] * normal
    cos_i = -np.abs(cos_i)
    # calculate new directions
    refl_dir_exp = rayDir - 2.0 * cos_i[:, None] * rayNormal
    k = 1.0 - eta**2 * (1.0 - cos_i**2)
    trans_dir_exp = eta[:, None] * rayDir
    with np.errstate(invalid="ignore"):
        # will produce NaN for total reflected rays (wanted)
        trans_dir_exp -= (eta * cos_i + np.sqrt(k))[:, None] * rayNormal

    # check results
    assert np.all(contrib > 0.0)
    assert np.allclose(refl_dir, refl_dir_exp, atol=1e-5)
    assert np.allclose(refl_contrib / contrib, r, atol=8e-4)
    assert np.allclose(trans_dir, trans_dir_exp, atol=2e-4, equal_nan=True)
    assert np.allclose(trans_contrib / contrib, t, atol=8e-4)

    # check polarization reference frame
    if polarized:
        # fetch polRef
        refl_polRef = structured_to_unstructured(polRef_buffer.numpy()["refl"])
        trans_polRef = structured_to_unstructured(polRef_buffer.numpy()["trans"])
        # check polRef are normal to plane of incidence
        # i.e. normal to both rayDir and surface normal
        assert np.abs(np.multiply(refl_polRef, rayDir).sum(-1)).max() < 5e-7
        assert np.abs(np.multiply(refl_polRef, normal).sum(-1)).max() < 5e-7
        assert np.abs(np.multiply(trans_polRef, rayDir).sum(-1)).max() < 5e-7
        assert np.abs(np.multiply(trans_polRef, normal).sum(-1)).max() < 5e-7

    # check stokes vector
    if polarized and forward:
        # fetch stokes vector
        refl_stokes = structured_to_unstructured(stokes_buffer.numpy()["refl"])
        trans_stokes = structured_to_unstructured(stokes_buffer.numpy()["trans"])
        # check stokes vector
        assert np.all(refl_stokes[:, 0] == 1.0)
        sr = (rp**2 - rs**2) / (rp**2 + rs**2)
        assert np.abs(refl_stokes[:, 1] - sr).max() < 1e-4
        assert np.all(refl_stokes[:, 2:] == 0.0)
        assert np.all(trans_stokes[:, 0] == 1.0)
        # bit larger error. Likely due to slight mismatch in refractive indices
        # GPU has linear interpolation, CPU uses analytic model
        st = (tp**2 - ts**2) / (tp**2 + ts**2)
        assert np.abs(trans_stokes[:, 1] - st).max() < 4e-3
        assert np.abs(trans_stokes[:, 1] - st).mean() < 1e-6
        assert np.all(trans_stokes[:, 2:] == 0.0)

    # check mueller matrix
    if polarized and not forward:
        # fetch mueller matrix
        refl_mueller = structured_to_unstructured(mueller_buffer.numpy()["refl"])
        trans_mueller = structured_to_unstructured(mueller_buffer.numpy()["trans"])
        # check mueller matrices
        m12 = (rp**2 - rs**2) / (rp**2 + rs**2)
        m33 = 2.0 * rp * rs / (rp**2 + rs**2)
        mueller_exp = np.zeros_like(refl_mueller)
        mueller_exp[:, 0] = 1.0
        mueller_exp[:, 1] = m12
        mueller_exp[:, 4] = m12
        mueller_exp[:, 5] = 1.0
        mueller_exp[:, 10] = m33
        mueller_exp[:, 15] = m33
        assert np.abs(refl_mueller - mueller_exp).max() < 5e-4
        m12 = (tp**2 - ts**2) / (tp**2 + ts**2)
        m33 = 2.0 * tp * ts / (tp**2 + ts**2)
        mueller_exp[:, 0] = 1.0
        mueller_exp[:, 1] = m12
        mueller_exp[:, 4] = m12
        mueller_exp[:, 5] = 1.0
        mueller_exp[:, 10] = m33
        mueller_exp[:, 15] = m33
        # TODO: Check why this error is so large
        assert np.abs(trans_mueller - mueller_exp).max() < 4e-3
        assert np.abs(trans_mueller - mueller_exp).mean() < 1e-7
