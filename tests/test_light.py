import numpy as np

import theia.light
import theia.items
import theia.material
import theia.random

from ctypes import *

from common.models import WaterModel

from hephaistos.queue import QueueTensor, as_queue
from hephaistos.pipeline import RetrieveTensorStage, runPipeline

from theia.util import packUint64


def test_lightsource(rng):
    N = 32 * 256
    N_PHOTONS = 4

    # create medium
    model = WaterModel()
    water = model.createMedium()
    tensor, _, media = theia.material.bakeMaterials(media=[water])

    # allocate queue
    item = theia.items.createRayQueueItem(N_PHOTONS)
    outputTensor = QueueTensor(item, N)

    # create light
    light = theia.light.HostLightSource(
        N, rayQueue=outputTensor, nPhotons=N_PHOTONS, medium=media["water"]
    )

    # fill input buffer
    rays = light.view(0)
    x = rng.random(N) * 10.0 - 5.0
    y = rng.random(N) * 10.0 - 5.0
    z = rng.random(N) * 10.0 - 5.0
    cos_theta = 2.0 * rng.random(N) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.random(N) * 2.0 * np.pi
    rays["position"] = np.stack([x, y, z], axis=-1)
    rays["direction"] = np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta],
        axis=-1,
    )
    lam = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    rays["wavelength"] = lam
    rays["startTime"] = rng.random((N, N_PHOTONS)) * 50.0
    rays["lin_contrib"] = rng.random((N, N_PHOTONS)) + 1.0
    rays["log_contrib"] = rng.random((N, N_PHOTONS)) * 5.0 - 4.0

    # run
    retrieve = RetrieveTensorStage(outputTensor)
    result = as_queue(retrieve.buffer(0), item)
    runPipeline([light, retrieve])

    # check result
    assert result.count == N
    assert np.allclose(result["position"], rays["position"])
    assert np.allclose(result["direction"], rays["direction"])
    assert np.all(result["rngStream"] == np.arange(N))
    assert np.all(result["rngCount"] == 0)
    water_packed = packUint64(media["water"])
    assert np.all(result["medium"] == (water_packed.x, water_packed.y))
    assert np.allclose(result["wavelength"], lam)
    assert np.allclose(result["time"], rays["startTime"])
    assert np.allclose(result["lin_contrib"], rays["lin_contrib"])
    assert np.allclose(result["log_contrib"], rays["log_contrib"])
    assert np.abs(result["n"] - model.refractive_index(lam)).max() < 5e-6
    assert np.abs(result["vg"] - model.group_velocity(lam)).max() < 1e-6
    assert np.abs(result["mu_s"] - model.scattering_coef(lam)).max() < 5e-5
    mu_e = model.scattering_coef(lam) + model.absorption_coef(lam)
    assert np.abs((result["mu_e"] - mu_e) / mu_e).max() < 4e-3


def test_sphericalLight():
    N = 32 * 256
    N_PHOTONS = 4
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
    item = theia.items.createRayQueueItem(N_PHOTONS)
    outputTensor = QueueTensor(item, N)

    # create light
    light = theia.light.SphericalLightSource(
        N,
        rayQueue=outputTensor,
        nPhotons=N_PHOTONS,
        position=position,
        rng=philox,
        lambdaRange=lamRange,
        timeRange=timeRange,
        intensity=intensity,
        medium=media["water"],
    )

    # run
    retrieve = RetrieveTensorStage(outputTensor)
    result = as_queue(retrieve.buffer(0), item)
    runPipeline([philox, light, retrieve])

    # check result
    assert result.count == N
    assert np.all(result["position"] == position)
    # uniform direction should average to zero
    assert np.abs(np.mean(result["direction"], axis=0)).max() < 0.01  # low statistics
    # correct rng indices
    assert np.all(result["rngStream"] == np.arange(N))
    assert np.all(result["rngCount"] == light.rngSamples)
    # check medium
    water_packed = packUint64(media["water"])
    assert np.all(result["medium"] == (water_packed.x, water_packed.y))
    # check contribution
    assert np.all(result["log_contrib"] == 0.0)
    assert np.all(result["lin_contrib"] == contrib)
    # lazily check "uniform" in time and lambda: check min/max
    assert np.abs(np.min(result["time"]) - timeRange[0]) < 1.0
    assert np.abs(np.max(result["time"]) - timeRange[1]) < 1.0
    assert np.abs(np.min(result["wavelength"]) - lamRange[0]) < 1.0
    assert np.abs(np.max(result["wavelength"]) - lamRange[1]) < 1.0
    # check constants
    lam = result["wavelength"].flatten()
    assert np.abs(result["n"].flatten() - model.refractive_index(lam)).max() < 5e-6
    assert np.abs(result["vg"].flatten() - model.group_velocity(lam)).max() < 1e-6
    assert np.abs(result["mu_s"].flatten() - model.scattering_coef(lam)).max() < 5e-5
    mu_e = model.scattering_coef(lam) + model.absorption_coef(lam)
    assert np.abs((result["mu_e"].flatten() - mu_e) / mu_e).max() < 4e-3
