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


def test_uniformWavelength():
    N = 32 * 256
    lamRange, dLam = (350.0, 750.0) * u.nm, 400.0 * u.nm
    timeRange, dt = (0.0, 0.0) * u.ns, 0.0 * u.ns
    # contrib = L/p(t,lam); p(t, lam) = 1.0 / (|dLam|*|dt|)
    contrib = dLam

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.UniformWavelengthSource(lambdaRange=lamRange)
    light = theia.light.PencilLightSource(photons, timeRange=timeRange)
    sampler = theia.light.LightSampler(light, N, rng=philox)
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.view(0)

    # check result
    assert np.all(result["contrib"] == contrib)
    assert np.abs(np.min(result["wavelength"]) - lamRange[0]) < 1.0
    assert np.abs(np.max(result["wavelength"]) - lamRange[1]) < 1.0


@pytest.mark.parametrize("polarized", [True, False])
def test_coneLightSource(polarized: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    direction = (0.8, 0.36, 0.48)
    opening = 0.33
    stokes = (1.0, 0.9, 0.1, -0.5)
    polRefIn = (0.0, 0.48, -0.36)
    budget = 12.0

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(100.0, 100.0) * u.nm)
    light = theia.light.ConeLightSource(
        photons,
        position=position,
        direction=direction,
        timeRange=(10.0, 10.0) * u.ns,
        cosOpeningAngle=opening,
        budget=budget,
        stokes=stokes,
        polarizationReference=polRefIn,
        polarized=polarized,
    )
    sampler = theia.light.LightSampler(light, N, rng=philox, polarized=polarized)
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.view(0)

    # check result
    assert result.count == N
    assert np.allclose(result["position"], position)
    assert (result["direction"] * direction).sum(-1).min() >= opening
    assert np.abs(np.square(result["direction"]).sum(-1) - 1.0).max() < 1e-5
    assert np.all(result["contrib"] == budget)
    if polarized:
        assert np.allclose(result["stokes"], stokes)
        polRef = result["polarizationRef"]
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-5
        # polRef is steepened to be perpendicular to ray dir
        # assert no rotation along ray direction happened, as this would make
        # the stokes vec wrong
        # no rotation happened if the parallelogram has zero volume
        vol = (np.cross(result["direction"], polRefIn) * polRef).sum(-1)
        assert np.abs(vol).max() < 1e-5


@pytest.mark.parametrize("polarized", [True, False])
def test_pencilLightSource(polarized: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    direction = (0.8, 0.36, 0.48)  # unit
    stokes = (1.0, 0.9, 0.1, -0.5)
    polRef = (0.0, 0.48, -0.36)
    budget = 12.0
    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(100.0, 100.0) * u.nm)
    light = theia.light.PencilLightSource(
        photons,
        position=position,
        direction=direction,
        timeRange=(10.0, 10.0) * u.ns,
        budget=budget,
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
    assert np.all(result["contrib"] == budget)
    if polarized:
        assert np.allclose(result["stokes"], stokes)
        assert np.allclose(result["polarizationRef"], polRef)


@pytest.mark.parametrize("polarized", [True, False])
def test_sphericalLightSource(polarized: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    budget = 12.0

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(100.0, 100.0) * u.nm)
    light = theia.light.SphericalLightSource(
        photons,
        position=position,
        timeRange=(10.0, 10.0) * u.ns,
        budget=budget,
    )
    sampler = theia.light.LightSampler(light, N, rng=philox, polarized=polarized)
    # run
    runPipeline([philox, photons, light, sampler])
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
    assert np.allclose(result["contrib"], budget)

    if polarized:
        assert np.allclose(result["stokes"], (1.0, 0.0, 0.0, 0.0))
        polRef = result["polarizationRef"]
        assert np.abs((polRef * result["direction"]).sum(-1)).max() < 1e-5
        assert (np.square(polRef).sum(-1) - 1.0).max() < 1e-5


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
    photons = theia.light.UniformWavelengthSource()
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
