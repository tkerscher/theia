import numpy as np
import hephaistos as hp

import theia.light
import theia.material
import theia.random

from ctypes import *

from common.items import Ray, N_PHOTONS
from common.models import WaterModel

from hephaistos.glsl import vec3, stackVector
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from theia.queue import QueueBuffer, QueueTensor, as_queue
from theia.scheduler import RetrieveTensorStage, runPipeline
from theia.util import packUint64


def test_lightsource(rng):
    N = 32 * 256

    # create medium
    model = WaterModel()
    water = model.createMedium()
    tensor, _, media = theia.material.bakeMaterials(media=[water])

    # allocate queue
    outputTensor = QueueTensor(Ray, N)

    # create light
    light = theia.light.HostLightSource(
        N, nBuffers=1, rayQueue=outputTensor, nPhotons=N_PHOTONS, medium=media["water"]
    )

    # fill input buffer
    rays = light.view(0)
    x = rng.random(N) * 10.0 - 5.0
    y = rng.random(N) * 10.0 - 5.0
    z = rng.random(N) * 10.0 - 5.0
    cos_theta = 2.0 * rng.random(N) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.random(N) * 2.0 * np.pi
    rays["position"] = stackVector((x, y, z), vec3)
    rays["direction"] = stackVector(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], vec3
    )
    rays["photons"]["wavelength"] = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    rays["photons"]["startTime"] = rng.random((N, N_PHOTONS)) * 50.0
    rays["photons"]["lin_contrib"] = rng.random((N, N_PHOTONS)) + 1.0
    rays["photons"]["log_contrib"] = rng.random((N, N_PHOTONS)) * 5.0 - 4.0

    # run
    retrieve = RetrieveTensorStage(outputTensor, 1)
    result = as_queue(retrieve.buffer(0), Ray)
    runPipeline([light, retrieve])

    # check result
    lam = rays["photons"]["wavelength"]
    assert result.count == N
    assert np.allclose(s2u(result["position"]), s2u(rays["position"]))
    assert np.allclose(s2u(result["direction"]), s2u(rays["direction"]))
    assert np.all(result["rngStream"] == np.arange(N))
    assert np.all(result["rngCount"] == 0)
    water_packed = packUint64(media["water"])
    assert np.all(result["medium"]["x"] == water_packed.x)
    assert np.all(result["medium"]["y"] == water_packed.y)
    photons = result["photons"]
    assert np.allclose(photons["wavelength"], lam)
    assert np.allclose(photons["time"], rays["photons"]["startTime"])
    assert np.allclose(photons["lin_c"], rays["photons"]["lin_contrib"])
    assert np.allclose(photons["log_c"], rays["photons"]["log_contrib"])
    assert np.abs(photons["n"] - model.refractive_index(lam)).max() < 5e-6
    assert np.abs(photons["vg"] - model.group_velocity(lam)).max() < 1e-6
    assert np.abs(photons["mu_s"] - model.scattering_coef(lam)).max() < 5e-5
    mu_e = model.scattering_coef(lam) + model.absorption_coef(lam)
    assert np.abs((photons["mu_e"] - mu_e) / mu_e).max() < 4e-3


def test_sphericalLight():
    N = 32 * 256
    position = (14.0, -2.0, 3.0)
    lamRange, dLam = (350.0, 750.0), 400.0
    timeRange, dt = (20.0, 70.0), 50.0
    intensity = 8.0
    contrib = intensity / dLam / dt

    # create rng
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)

    # create medium
    model = WaterModel()
    water = model.createMedium()
    tensor, _, media = theia.material.bakeMaterials(media=[water])

    # allocate queue
    outputTensor = QueueTensor(Ray, N)

    # create light
    light = theia.light.SphericalLightSource(
        N,
        rayQueue=outputTensor,
        nPhotons=N_PHOTONS,
        position=position,
        rng=philox,
        wavelengthRange=lamRange,
        timeRange=timeRange,
        intensity=intensity,
        medium=media["water"],
    )

    # run
    retrieve = RetrieveTensorStage(outputTensor, 1)
    result = as_queue(retrieve.buffer(0), Ray)
    runPipeline([philox, light, retrieve])

    # check result
    assert result.count == N
    photons = result["photons"]
    pos = s2u(result["position"])
    assert np.all(pos == position)
    # uniform direction should average to zero
    dir = s2u(result["direction"])
    assert np.abs(np.mean(dir, axis=0)).max() < 0.01  # low statistics
    # correct rng indices
    assert np.all(result["rngStream"] == np.arange(N))
    assert np.all(result["rngCount"] == light.rngSamples)
    # check medium
    water_packed = packUint64(media["water"])
    assert np.all(result["medium"]["x"] == water_packed.x)
    assert np.all(result["medium"]["y"] == water_packed.y)
    # check contribution
    assert np.all(photons["log_c"] == 0.0)
    assert np.all(photons["lin_c"] == contrib)
    # lazily check "uniform" in time and lambda: check min/max
    assert np.abs(np.min(photons["time"]) - timeRange[0]) < 1.0
    assert np.abs(np.max(photons["time"]) - timeRange[1]) < 1.0
    assert np.abs(np.min(photons["wavelength"]) - lamRange[0]) < 1.0
    assert np.abs(np.max(photons["wavelength"]) - lamRange[1]) < 1.0
    # check constants
    lam = photons["wavelength"].flatten()
    assert np.abs(photons["n"].flatten() - model.refractive_index(lam)).max() < 5e-6
    assert np.abs(photons["vg"].flatten() - model.group_velocity(lam)).max() < 1e-6
    assert np.abs(photons["mu_s"].flatten() - model.scattering_coef(lam)).max() < 5e-5
    mu_e = model.scattering_coef(lam) + model.absorption_coef(lam)
    assert np.abs((photons["mu_e"].flatten() - mu_e) / mu_e).max() < 4e-3
