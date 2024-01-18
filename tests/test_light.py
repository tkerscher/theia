import numpy as np

import theia.light
import theia.items
import theia.material
import theia.random

from ctypes import *

from hephaistos.pipeline import runPipeline


def test_lightsource(rng):
    N = 32 * 256
    N_LAMBDA = 4

    # create light & sampler
    light = theia.light.HostLightSource(N, nLambda=N_LAMBDA)
    sampler = theia.light.LightSampler(light, N)

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
    lam = rng.random((N, N_LAMBDA)) * 600.0 + 200.0
    rays["wavelength"] = lam
    rays["startTime"] = rng.random((N, N_LAMBDA)) * 50.0
    rays["contrib"] = rng.random((N, N_LAMBDA)) + 1.0

    # run
    runPipeline([light, sampler])
    result = sampler.view(0)

    # check result
    assert result.count == N
    assert np.allclose(result["position"], rays["position"])
    assert np.allclose(result["direction"], rays["direction"])
    assert np.allclose(result["wavelength"], lam)
    assert np.allclose(result["startTime"], rays["startTime"])
    assert np.allclose(result["contrib"], rays["contrib"])


def test_diskRay():
    N = 32 * 1024
    center = (14.0, -2.0, 3.0)
    direction = (0.8, 0.36, 0.48)  # unit
    radius = 5.0

    # create pipeline
    philox = theia.random.SobolQRNG(seed=0xC0110FFC0FFEE)
    rays = theia.light.DiskRaySource(center=center, direction=direction, radius=radius)
    photons = theia.light.UniformPhotonSource(
        lambdaRange=(100.0, 100.0), timeRange=(10.0, 10.0)
    )
    light = theia.light.ModularLightSource(rays, photons, 1)
    sampler = theia.light.LightSampler(light, N, rng=philox)
    # run
    runPipeline([philox, rays, photons, light, sampler])
    result = sampler.view(0)

    # transform input to numpy array for testing
    center = np.array(center, dtype=np.float32)
    direction = np.array(direction, dtype=np.float32)
    # check result
    assert result.count == N
    # check position to center is perpendicular to direction -> on disk plane
    v = result["position"] - center
    assert np.abs(np.multiply(v, direction).sum(-1)).max() < 1e-5
    # check distance to center
    dist = np.sqrt(np.square(v).sum(-1))
    assert np.abs(dist.max() - radius) < 5e-4
    assert dist.min() < 0.05  # TODO: why so large?
    # not enough samples for histogram -> check mean is 2/3 the radius
    assert np.abs(3.0 * dist.mean() - 2.0 * radius) < 1e-3
    # check contrib of 1/area
    assert np.allclose(result["contrib"], np.pi * radius**2)


def test_pencilRay():
    N = 32 * 256
    position = (14.0, -2.0, 3.0)
    direction = (0.8, 0.36, 0.48)  # unit

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    rays = theia.light.PencilRaySource(position=position, direction=direction)
    photons = theia.light.UniformPhotonSource(
        lambdaRange=(100.0, 100.0), timeRange=(10.0, 10.0)
    )
    light = theia.light.ModularLightSource(rays, photons, 1)
    sampler = theia.light.LightSampler(light, N, rng=philox)
    # run
    runPipeline([philox, rays, photons, light, sampler])
    result = sampler.view(0)

    # transform input to numpy array for testing
    position = np.array(position, dtype=np.float32)
    direction = np.array(direction, dtype=np.float32)
    # check result
    assert result.count == N
    assert np.all(result["position"] == position)
    assert np.all(result["direction"] == direction)
    assert np.all(result["contrib"] == 1.0)


def test_sphericalRay():
    N = 32 * 256
    position = (14.0, -2.0, 3.0)

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    rays = theia.light.SphericalRaySource(position)
    photons = theia.light.UniformPhotonSource(
        lambdaRange=(100.0, 100.0), timeRange=(10.0, 10.0)
    )
    light = theia.light.ModularLightSource(rays, photons, 1)
    sampler = theia.light.LightSampler(light, N, rng=philox)
    # run
    runPipeline([philox, rays, photons, light, sampler])
    result = sampler.view(0)

    # transform input to numpy array for testing
    position = np.array(position, dtype=np.float32)
    # check result
    assert result.count == N
    assert np.all(result["position"] == position)
    assert np.allclose(np.sqrt(np.square(result["direction"]).sum(-1)), 1.0)
    # uniform direction should average to zero
    assert np.abs(np.mean(result["direction"], axis=0)).max() < 0.01  # low statistics
    # check contribution
    assert np.allclose(result["contrib"], 4.0 * np.pi)


def test_uniformPhoton():
    N = 32 * 256
    N_LAMBDA = 4
    lamRange, dLam = (350.0, 750.0), 400.0
    timeRange, dt = (20.0, 70.0), 50.0
    intensity = 8.0
    contrib = intensity / dLam / dt

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    rays = theia.light.PencilRaySource()
    photons = theia.light.UniformPhotonSource(
        lambdaRange=lamRange, timeRange=timeRange, intensity=intensity
    )
    light = theia.light.ModularLightSource(rays, photons, N_LAMBDA)
    sampler = theia.light.LightSampler(light, N, rng=philox)
    # run
    runPipeline([philox, rays, photons, light, sampler])
    result = sampler.view(0)

    # check result
    assert np.all(result["contrib"] == contrib)
    assert np.abs(np.min(result["startTime"]) - timeRange[0]) < 1.0
    assert np.abs(np.max(result["startTime"]) - timeRange[1]) < 1.0
    assert np.abs(np.min(result["wavelength"]) - lamRange[0]) < 1.0
    assert np.abs(np.max(result["wavelength"]) - lamRange[1]) < 1.0
