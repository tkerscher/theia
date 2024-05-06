import pytest

import numpy as np
import scipy.constants as c
from hephaistos.pipeline import runPipeline
import theia.units as u

import theia.light
import theia.material
import theia.random

from ctypes import *

from .common.models import WaterModel


@pytest.mark.parametrize("polarized", [True, False])
def test_hostLightSource(rng, polarized: bool):
    N = 32 * 256

    # create light & sampler
    light = theia.light.HostLightSource(N, polarized=polarized)
    sampler = theia.light.LightSampler(light, N, polarized=polarized)

    # fill input buffer
    rays = light.view(0)
    x = (rng.random(N) * 10.0 - 5.0) * u.m
    y = (rng.random(N) * 10.0 - 5.0) * u.m
    z = (rng.random(N) * 10.0 - 5.0) * u.m
    cos_theta = 2.0 * rng.random(N) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.random(N) * 2.0 * np.pi
    rays["position"] = np.stack([x, y, z], axis=-1)
    rays["direction"] = np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta],
        axis=-1,
    )
    lam = (rng.random((N,)) * 600.0 + 200.0) * u.nm
    rays["wavelength"] = lam
    rays["startTime"] = rng.random((N,)) * 50.0 * u.ns
    rays["contrib"] = rng.random((N,)) + 1.0
    if polarized:
        # We dont care about valid values
        rays["stokes"] = rng.random((N, 4))
        rays["polarizationRef"] = rng.random((N, 3))

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
    if polarized:
        assert np.allclose(result["stokes"], rays["stokes"])
        assert np.allclose(result["polarizationRef"], rays["polarizationRef"])


def test_HostSamplerMismatch():
    with pytest.raises(RuntimeError):
        light = theia.light.HostLightSource(128, polarized=True)
        sampler = theia.light.LightSampler(light, 128, polarized=False)
    with pytest.raises(RuntimeError):
        light = theia.light.HostLightSource(128, polarized=False)
        sampler = theia.light.LightSampler(light, 128, polarized=True)


def test_diskRay():
    N = 32 * 1024
    center = (14.0, -2.0, 3.0) * u.m
    direction = (0.8, 0.36, 0.48)  # unit
    radius = 5.0 * u.m

    # create pipeline
    philox = theia.random.SobolQRNG(seed=0xC0110FFC0FFEE)
    rays = theia.light.DiskRaySource(center=center, direction=direction, radius=radius)
    photons = theia.light.UniformPhotonSource(
        lambdaRange=(100.0, 100.0) * u.nm, timeRange=(10.0, 10.0) * u.ns
    )
    light = theia.light.ModularLightSource(rays, photons)
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


@pytest.mark.parametrize("polarized", [True, False])
def test_pencilRay(polarized: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    direction = (0.8, 0.36, 0.48)  # unit

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    rays = theia.light.PencilRaySource(position=position, direction=direction)
    photons = theia.light.UniformPhotonSource(
        lambdaRange=(100.0, 100.0) * u.nm, timeRange=(10.0, 10.0) * u.ns
    )
    light = theia.light.ModularLightSource(rays, photons)
    sampler = theia.light.LightSampler(light, N, rng=philox, polarized=polarized)
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
    if polarized:
        assert np.allclose(result["stokes"], (1.0, 0.0, 0.0, 0.0))
        # we only require polRef to be unit and perpendicular to direction
        dir = np.array(direction)
        polRef = result["polarizationRef"]
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        assert np.abs((polRef * dir[None, :]).sum(-1)).max() < 1e-6


def test_sphericalRay():
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    rays = theia.light.SphericalRaySource(position)
    photons = theia.light.UniformPhotonSource(
        lambdaRange=(100.0, 100.0) * u.nm, timeRange=(10.0, 10.0) * u.ns
    )
    light = theia.light.ModularLightSource(rays, photons)
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
    lamRange, dLam = (350.0, 750.0) * u.nm, 400.0 * u.nm
    timeRange, dt = (20.0, 70.0) * u.ns, 50.0 * u.ns
    budget = 8.0
    # contrib = L/p(t,lam); p(t, lam) = 1.0 / (|dLam|*|dt|)
    contrib = budget * dLam * dt

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    rays = theia.light.PencilRaySource()
    photons = theia.light.UniformPhotonSource(
        lambdaRange=lamRange, timeRange=timeRange, budget=budget
    )
    light = theia.light.ModularLightSource(rays, photons)
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


@pytest.mark.parametrize("polarized", [True, False])
def test_pencilLightSource(rng, polarized: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    direction = (0.8, 0.36, 0.48)  # unit
    stokes = (1.0, 0.9, 0.1, -0.5)
    polRef = (0.0, 0.48, -0.36)

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.UniformPhotonSource(
        lambdaRange=(100.0, 100.0) * u.nm, timeRange=(10.0, 10.0) * u.ns
    )
    light = theia.light.PencilLightSource(
        photons,
        position=position,
        direction=direction,
        stokes=stokes,
        polarizationRef=polRef,
    )
    sampler = theia.light.LightSampler(light, N, rng=philox, polarized=polarized)
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.view(0)

    # check result
    assert result.count == N
    assert np.allclose(result["position"], position)
    assert np.allclose(result["direction"], direction)
    assert np.all(result["contrib"] == 1.0)
    if polarized:
        assert np.allclose(result["stokes"], stokes)
        assert np.allclose(result["polarizationRef"], polRef)


@pytest.mark.parametrize("usePhotons", [False, True])
@pytest.mark.parametrize("polarized", [True, False])
def test_cherenkovTrack(usePhotons: bool, polarized: bool):
    N = 32 * 256
    vertices = np.array(
        [
            #  x,  y,  z, t
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 20.0],
            [1.0, 1.0, 0.0, 35.0],
            [1.0, 1.0, 1.0, 60.0],
        ]
    )
    trackDir = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    dt = np.array([20.0, 15.0, 25.0])

    # build media
    model = WaterModel()
    water = model.createMedium()
    store = theia.material.MaterialStore([], media=[water])

    # build track
    track = theia.light.ParticleTrack(4)
    track.setVertices(vertices)
    # build light source
    photons = theia.light.UniformPhotonSource(timeRange=(0.0, 0.0))
    light = theia.light.CherenkovTrackLightSource(
        photons,
        track,
        medium=store.media["water"],
        usePhotonCount=usePhotons,
    )
    # build pipeline
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    sampler = theia.light.LightSampler(light, N, rng=philox, polarized=polarized)
    # run pipeline
    runPipeline([philox, photons, light, sampler])

    # check result
    samples = sampler.view(0)
    assert np.all((samples["position"] >= 0.0) & (samples["position"] <= 1.0))
    t = samples["startTime"].ravel()
    assert np.all((t >= 0.0) & (t <= 60.0))
    t_exp = np.multiply(samples["position"], dt[None, :]).sum(-1)
    assert np.allclose(t, t_exp)
    segmentId = 2 - (samples["position"] == 0.0).sum(-1)
    cos_theta = np.multiply(samples["direction"], trackDir[segmentId]).sum(-1)
    lam = samples["wavelength"].ravel()
    n = model.refractive_index(lam)
    assert np.allclose(cos_theta, 1.0 / n)
    contrib = None
    if usePhotons:
        ft_const = 2.0 * c.pi * c.alpha * 1e9
        # frank tamm; 3.0 from segment sampling
        contrib = ft_const / (lam**2) * (1.0 - (1.0 / n**2)) * 3.0
    else:
        ft_const = c.pi * c.e * c.c**2 * c.mu_0 * 1e18
        contrib = ft_const / (lam**3) * (1.0 - (1.0 / n**2)) * 3.0
    # contrib has additional factor from wavelength sampling
    lam0, lam1 = photons.getParam("lambdaRange")
    contrib *= abs(lam1 - lam0)
    assert np.allclose(contrib, samples["contrib"].ravel())
    if polarized:
        polRef_exp = np.cross(trackDir[segmentId], samples["direction"])
        polRef_exp /= np.sqrt(np.square(polRef_exp).sum(-1))[:, None]
        polRef = samples["polarizationRef"]
        assert np.abs(np.abs((polRef * polRef_exp).sum(-1)) - 1.0).max() < 1e-6
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        assert np.abs((polRef * samples["direction"]).sum(-1)).max() < 1e-6
