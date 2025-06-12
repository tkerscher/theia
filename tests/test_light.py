import pytest

from dataclasses import asdict
import numpy as np
import scipy.constants as c
import scipy.integrate as integrate
import scipy.stats as stats
from scipy.stats.sampling import NumericalInversePolynomial

import hephaistos as hp
from hephaistos.pipeline import runPipeline
from ctypes import Structure, c_float

import theia.cascades
import theia.light
from theia.material import MaterialStore
from theia.random import PhiloxRNG
from theia.testing import BackwardLightSampler, WaterTestModel
import theia.units as u


@pytest.mark.parametrize("polarized", [True, False])
def test_hostLightSource(rng, polarized: bool):
    N = 32 * 256

    # create light & sampler
    photon = theia.light.HostWavelengthSource(N)
    light = theia.light.HostLightSource(N, polarized=polarized)
    sampler = theia.light.LightSampler(light, photon, N, polarized=polarized)

    # fill input buffer
    rays = light.queue.view(0)
    photons = photon.queue.view(0)
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
    photons["wavelength"] = lam
    photons["contrib"] = rng.random((N,)) + 5.0
    rays["startTime"] = rng.random((N,)) * 50.0 * u.ns
    rays["contrib"] = rng.random((N,)) + 1.0
    if polarized:
        # We dont care about valid values
        rays["stokes"] = rng.random((N, 4))
        rays["polarizationRef"] = rng.random((N, 3))

    # run
    runPipeline([photon, light, sampler])
    lightResult = sampler.lightQueue.view(0)
    phResult = sampler.wavelengthQueue.view(0)

    # check result
    assert lightResult.count == N
    assert np.allclose(lightResult["position"], rays["position"])
    assert np.allclose(lightResult["direction"], rays["direction"])
    assert np.allclose(lightResult["startTime"], rays["startTime"])
    assert np.allclose(lightResult["contrib"], rays["contrib"])
    assert np.allclose(phResult["contrib"], photons["contrib"])
    assert np.allclose(phResult["wavelength"], lam)
    if polarized:
        assert np.allclose(lightResult["stokes"], rays["stokes"])
        assert np.allclose(lightResult["polarizationRef"], rays["polarizationRef"])


def test_HostSamplerMismatch():
    photon = theia.light.ConstWavelengthSource()
    with pytest.raises(RuntimeError):
        light = theia.light.HostLightSource(128, polarized=True)
        sampler = theia.light.LightSampler(light, photon, 128, polarized=False)
    with pytest.raises(RuntimeError):
        light = theia.light.HostLightSource(128, polarized=False)
        sampler = theia.light.LightSampler(light, photon, 128, polarized=True)


def test_constWavelength():
    N = 32 * 256
    lam = 450.0 * u.nm

    # create pipeline
    philox = PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(lam)
    light = theia.light.PencilLightSource(timeRange=(0.0, 0.0))
    sampler = theia.light.LightSampler(light, photons, N, rng=philox)
    # run
    runPipeline([photons, light, sampler])
    result = sampler.wavelengthQueue.view(0)

    # check result
    assert np.all(result["contrib"] == 1.0)
    assert np.all(result["wavelength"] == lam)


@pytest.mark.parametrize("normalize", [True, False])
def test_uniformWavelength(normalize: bool):
    N = 32 * 256
    lamRange, dLam = (350.0, 750.0) * u.nm, 400.0 * u.nm
    timeRange, dt = (0.0, 0.0) * u.ns, 0.0 * u.ns
    # contrib = L/p(t,lam); p(t, lam) = 1.0 / (|dLam|*|dt|)
    contrib = 1.0 if normalize else dLam

    # create pipeline
    philox = PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.UniformWavelengthSource(
        lambdaRange=lamRange, normalize=normalize
    )
    light = theia.light.PencilLightSource(timeRange=timeRange)
    sampler = theia.light.LightSampler(light, photons, N, rng=philox)
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.wavelengthQueue.view(0)

    # check result
    assert np.all(result["contrib"] == contrib)
    assert np.abs(np.min(result["wavelength"]) - lamRange[0]) < 1.0
    assert np.abs(np.max(result["wavelength"]) - lamRange[1]) < 1.0


def test_functionWavelength():
    N = 256 * 1024  # a bit more for the histogram
    lamRange = (300.0, 700.0) * u.nm

    def fn(lam: float) -> float:
        # something fancy just for testing
        x = lam / 1000
        return -x * np.log(x)

    def Fn(lam: float) -> float:
        # anti-derivative needed for testing
        x = lam / 1000
        return 250.0 * x**2 * (1.0 - 2.0 * np.log(x))

    # create pipeline
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.FunctionWavelengthSource(fn, lambdaRange=lamRange)
    light = theia.light.PencilLightSource(timeRange=(0.0, 0.0))
    sampler = theia.light.LightSampler(light, photons, N, rng=philox)
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.wavelengthQueue.view(0)

    # check result
    exp_contrib = Fn(lamRange[1]) - Fn(lamRange[0])
    assert np.allclose(result["contrib"], exp_contrib)
    assert np.abs(np.min(result["wavelength"]) - lamRange[0]) < 1.0
    assert np.abs(np.max(result["wavelength"]) - lamRange[1]) < 1.0
    # check distribution via histogram
    hist, edges = np.histogram(result["wavelength"], bins=40)
    hist = hist / N
    F = Fn(edges)
    exp_hist = (F[1:] - F[:-1]) / exp_contrib
    assert np.abs(hist - exp_hist).max() < 9e-4


@pytest.mark.parametrize("polarized", [True, False])
def test_coneLightSource_fwd(polarized: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    direction = (0.8, 0.36, 0.48)
    opening = 0.33
    stokes = (1.0, 0.9, 0.1, -0.5)
    polRefIn = (0.0, 0.48, -0.36)
    budget = 12.0

    # create pipeline
    philox = PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(wavelength=100.0 * u.nm)
    light = theia.light.ConeLightSource(
        position=position,
        direction=direction,
        timeRange=(10.0, 10.0) * u.ns,
        cosOpeningAngle=opening,
        budget=budget,
        stokes=stokes,
        polarizationReference=polRefIn,
        polarized=polarized,
    )
    sampler = theia.light.LightSampler(
        light, photons, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.lightQueue.view(0)

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
def test_coneLightSource_bwd(polarized: bool):
    N = 32 * 256
    light_pos = (14.0, -2.0, 3.0) * u.m
    light_dir = (0.8, 0.36, 0.48)
    light_opening = 0.33
    light_stokes = (1.0, 0.9, 0.1, -0.5)
    polRefIn = (0.0, 0.48, -0.36)
    budget = 12.0

    # create pipeline
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.ConstWavelengthSource()
    light = theia.light.ConeLightSource(
        position=light_pos,
        direction=light_dir,
        timeRange=(10.0, 10.0) * u.ns,
        cosOpeningAngle=light_opening,
        budget=budget,
        stokes=light_stokes,
        polarizationReference=polRefIn,
        polarized=polarized,
    )
    sampler = BackwardLightSampler(N, light, photons, rng=philox, polarized=polarized)
    # run
    runPipeline(sampler.collectStages())
    result = sampler.getResults(0)

    light_pos = np.array(light_pos)
    light_dir = np.array(light_dir)
    # retrieve results
    startTime = result["startTime"]
    contrib = result["contrib"]
    # check result
    assert np.isfinite(result["observer"]).all()
    assert np.all(result["position"] == light_pos)
    exp_dir = result["observer"] - light_pos[None, :]
    d = np.sqrt(np.square(exp_dir).sum(-1))
    exp_dir /= d[:, None]
    assert np.allclose(result["direction"], exp_dir)
    assert np.all(startTime == 10.0 * u.ns)
    cos_nrm = np.abs(np.multiply(result["normal"], result["direction"]).sum(-1))
    cos_angle = np.multiply(light_dir[None, :], result["direction"]).sum(-1)
    exp_contrib = budget * cos_nrm / ((1.0 - light_opening) * 2.0 * np.pi * d**2)
    exp_contrib = np.where(cos_angle >= light_opening, exp_contrib, 0.0)
    assert np.allclose(contrib, exp_contrib)
    # tests for polarization
    if polarized:
        assert np.allclose(result["stokes"], light_stokes)
        assert np.abs(np.square(result["polRef"]).sum(-1) - 1.0).max() < 1e-5
        # polRef is steepened to be perpendicular to ray dir
        # assert no rotation along ray direction happened, as this would make
        # the stokes vec wrong
        # no rotation happened if the parallelogram has zero volume
        vol = (np.cross(result["direction"], polRefIn) * result["polRef"]).sum(-1)
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
    philox = PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(wavelength=100.0 * u.nm)
    light = theia.light.PencilLightSource(
        position=position,
        direction=direction,
        timeRange=(10.0, 10.0) * u.ns,
        budget=budget,
        stokes=stokes,
        polarizationRef=polRef,
    )
    sampler = theia.light.LightSampler(
        light, photons, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.lightQueue.view(0)

    # check result
    assert result.count == N
    assert np.allclose(result["position"], position)
    assert np.allclose(result["direction"], direction)
    assert np.all(result["contrib"] == budget)
    if polarized:
        assert np.allclose(result["stokes"], stokes)
        assert np.allclose(result["polarizationRef"], polRef)


@pytest.mark.parametrize("polarized", [True, False])
def test_sphericalLightSource_fwd(polarized: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    budget = 12.0

    # create pipeline
    philox = PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(wavelength=100.0 * u.nm)
    light = theia.light.SphericalLightSource(
        position=position,
        timeRange=(10.0, 10.0) * u.ns,
        budget=budget,
    )
    sampler = theia.light.LightSampler(
        light, photons, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.lightQueue.view(0)

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


@pytest.mark.parametrize("polarized", [True, False])
def test_sphericalLightSource_bwd(polarized: bool):
    N = 32 * 256
    light_pos = (14.0, -2.0, 3.0) * u.m
    budget = 1e6

    # create pipeline
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.ConstWavelengthSource()
    light = theia.light.SphericalLightSource(
        position=light_pos,
        timeRange=(10.0, 10.0) * u.ns,
        budget=budget,
    )
    sampler = BackwardLightSampler(N, light, photons, rng=philox, polarized=polarized)
    # run
    runPipeline(sampler.collectStages())
    result = sampler.getResults(0)

    # check result
    light_pos = np.array(light_pos)
    assert np.isfinite(result["observer"]).all()
    assert np.all(result["position"] == light_pos)
    exp_dir = result["observer"] - light_pos[None, :]
    d = np.sqrt(np.square(exp_dir).sum(-1))
    exp_dir /= d[:, None]
    assert np.allclose(result["direction"], exp_dir)
    assert np.all(result["startTime"] == 10.0 * u.ns)
    cos_nrm = np.abs(np.multiply(result["normal"], result["direction"]).sum(-1))
    exp_contrib = budget * cos_nrm / (4.0 * np.pi * d**2)
    # bit larger error, likely precission issues (float vs double)
    assert np.allclose(result["contrib"], exp_contrib, atol=1e-5)
    # tests for polarization
    if polarized:
        assert np.allclose(result["stokes"], (1.0, 0.0, 0.0, 0.0))
        assert np.abs((result["polRef"] * result["direction"]).sum(-1)).max() < 1e-5
        assert (np.square(result["polRef"]).sum(-1) - 1.0).max() < 1e-5


@pytest.mark.parametrize("usePhotons", [False, True])
@pytest.mark.parametrize("polarized", [True, False])
def test_cherenkov_fwd(usePhotons: bool, polarized: bool):
    N = 32 * 256
    dir = np.array([0.36, 0.48, 0.8])
    dist = 200.0 * u.m
    startPos, startT = -dir * 0.5 * dist, -0.5 * dist / u.c
    endPos, endT = dir * 0.5 * dist, 0.5 * dist / u.c
    # build media
    model = WaterTestModel()
    water = model.createMedium()
    store = MaterialStore([], media=[water])

    # build pipeline
    philox = PhiloxRNG(key=0xC0FFEE)
    photon = theia.light.UniformWavelengthSource(normalize=False)
    light = theia.light.CherenkovLightSource(
        trackStart=startPos,
        trackEnd=endPos,
        startTime=startT,
        endTime=endT,
        usePhotonCount=usePhotons,
    )
    sampler = theia.light.LightSampler(
        light, photon, N, rng=philox, medium=store.media["water"], polarized=polarized
    )
    # run
    runPipeline([philox, photon, light, sampler])
    samples = sampler.lightQueue.view(0)
    photons = sampler.wavelengthQueue.view(0)

    # check result
    distA = np.sqrt(np.square(samples["position"] - startPos).sum(-1))
    distB = np.sqrt(np.square(samples["position"] - endPos).sum(-1))
    assert np.allclose(distA + distB, dist)  # check position is on track
    t = samples["startTime"].ravel()
    assert np.all((t >= startT) & (t <= endT))
    l = distA / dist
    t_exp = (1.0 - l) * startT + l * endT
    assert np.allclose(t, t_exp)
    cos_theta = np.multiply(samples["direction"], dir[None, :]).sum(-1)
    lam = photons["wavelength"]
    n = model.refractive_index(lam)
    assert np.allclose(cos_theta, 1.0 / n)
    contrib = 1.0 - (1.0 / n**2)
    if usePhotons:
        ft_const = 2.0 * np.pi * c.alpha * 1e9
        contrib *= ft_const / (lam**2)
    else:
        ft_const = np.pi * c.e * c.c**2 * c.mu_0 * 1e18
        contrib *= ft_const / (lam**3)
    lam0, lam1 = photon.lambdaRange
    contrib *= dist  # 1/p(lam) 1/p(pos)
    assert np.allclose(contrib, samples["contrib"].ravel())
    assert np.allclose(photons["contrib"], abs(lam1 - lam0))
    # check polarization
    if polarized:
        polRef_exp = np.cross(dir, samples["direction"])
        polRef_exp /= np.sqrt(np.square(polRef_exp).sum(-1))[:, None]
        polRef = samples["polarizationRef"]
        assert np.allclose(samples["stokes"], (1.0, 1.0, 0.0, 0.0))
        assert np.abs(np.abs((polRef * polRef_exp).sum(-1)) - 1.0).max() < 1e-6
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        assert np.abs((polRef * samples["direction"]).sum(-1)).max() < 1e-6


@pytest.mark.parametrize("usePhotons", [False, True])
@pytest.mark.parametrize("polarized", [True, False])
def test_cherenkov_bwd(usePhotons: bool, polarized: bool):
    N = 32 * 256
    lam_range = (400.0, 800.0) * u.nm
    dir = np.array([0.36, 0.48, 0.8])
    # dir = np.array([1.0, 0.0, 0.0])
    dist = 200.0 * u.m
    startPos, startT = -dir * 0.5 * dist, -0.5 * dist / u.c
    endPos, endT = dir * 0.5 * dist, 0.5 * dist / u.c
    # build media
    model = WaterTestModel()
    water = model.createMedium()
    store = MaterialStore([], media=[water])

    # build pipeline
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.UniformWavelengthSource(lambdaRange=lam_range)
    light = theia.light.CherenkovLightSource(
        trackStart=startPos,
        trackEnd=endPos,
        startTime=startT,
        endTime=endT,
        usePhotonCount=usePhotons,
    )
    sampler = BackwardLightSampler(
        N,
        light,
        photons,
        rng=philox,
        medium=store.media["water"],
        polarized=polarized,
    )
    # run
    runPipeline(sampler.collectStages())
    result = sampler.getResults(0)

    # retrieve results
    startTime = result["startTime"]
    contrib = result["contrib"]
    mask = contrib != 0.0
    # check results
    assert np.isfinite(result["observer"]).all()
    distA = np.sqrt(np.square(result["position"] - startPos).sum(-1))
    distB = np.sqrt(np.square(result["position"] - endPos).sum(-1))
    assert np.allclose((distA + distB)[mask], dist)  # check position is on track
    assert np.all((startTime[mask] >= startT) & (startTime[mask] <= endT))
    l = distA / dist
    t_exp = (1.0 - l) * startT + l * endT
    assert np.allclose(startTime[mask], t_exp[mask], atol=5e-5)
    exp_dir = result["observer"] - result["position"]
    d = np.sqrt(np.square(exp_dir).sum(-1))
    exp_dir /= d[:, None]
    assert np.allclose(result["direction"], exp_dir)
    cos_theta = np.multiply(result["direction"], dir[None, :]).sum(-1)
    n = model.refractive_index(result["wavelength"])
    assert np.allclose(cos_theta, 1.0 / n)
    exp_contrib = 1.0 - (1.0 / n**2)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    cos_nrm = np.abs(np.multiply(result["direction"], result["normal"]).sum(-1))
    exp_contrib *= cos_nrm / (sin_theta * d)
    if usePhotons:
        ft_const = c.alpha * 1e9
        exp_contrib *= ft_const / (result["wavelength"] ** 2)
    else:
        ft_const = 0.5 * c.e * c.c**2 * c.mu_0 * 1e18
        exp_contrib *= ft_const / (result["wavelength"] ** 3)
    assert np.allclose(contrib[mask], exp_contrib[mask])
    # check polarization
    if polarized:
        stokes = result["stokes"]
        polRef = result["polRef"]
        assert np.allclose(stokes, (1.0, 1.0, 0.0, 0.0))
        polRef_exp = np.cross(dir, result["direction"])
        polRef_exp /= np.sqrt(np.square(polRef_exp).sum(-1))[:, None]
        assert np.abs(np.abs((polRef * polRef_exp).sum(-1)) - 1.0).max() < 1e-6
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        assert np.abs((polRef * result["direction"]).sum(-1)).max() < 1e-6


@pytest.mark.parametrize("usePhotons", [False, True])
@pytest.mark.parametrize("polarized", [True, False])
def test_cherenkovTrack(usePhotons: bool, polarized: bool):
    N = 32 * 256
    # fmt: off
    vertices = np.array([
        #  x,  y,  z, t
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 20.0],
        [1.0, 1.0, 0.0, 35.0],
        [1.0, 1.0, 1.0, 60.0],
    ])
    trackDir = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    # fmt: on
    dt = np.array([20.0, 15.0, 25.0])

    # build media
    model = WaterTestModel()
    water = model.createMedium()
    store = MaterialStore([], media=[water])

    # build track
    track = theia.light.ParticleTrack(4)
    track.setVertices(vertices)
    # build light source
    photon = theia.light.UniformWavelengthSource(normalize=False)
    light = theia.light.CherenkovTrackLightSource(
        track,
        usePhotonCount=usePhotons,
    )
    # build pipeline
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = theia.light.LightSampler(
        light, photon, N, rng=philox, medium=store.media["water"], polarized=polarized
    )
    # run pipeline
    runPipeline([philox, photon, light, sampler])
    lam0, lam1 = photon.getParam("lambdaRange")
    samples = sampler.lightQueue.view(0)
    photons = sampler.wavelengthQueue.view(0)

    # check result
    assert np.all((samples["position"] >= 0.0) & (samples["position"] <= 1.0))
    t = samples["startTime"].ravel()
    assert np.all((t >= 0.0) & (t <= 60.0))
    t_exp = np.multiply(samples["position"], dt[None, :]).sum(-1)
    assert np.allclose(t, t_exp)
    segmentId = 2 - (samples["position"] == 0.0).sum(-1)
    cos_theta = np.multiply(samples["direction"], trackDir[segmentId]).sum(-1)
    lam = photons["wavelength"].ravel()
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
    assert np.allclose(abs(lam1 - lam0), photons["contrib"].ravel())
    assert np.allclose(contrib, samples["contrib"].ravel())
    if polarized:
        polRef_exp = np.cross(trackDir[segmentId], samples["direction"])
        polRef_exp /= np.sqrt(np.square(polRef_exp).sum(-1))[:, None]
        polRef = samples["polarizationRef"]
        assert np.abs(np.abs((polRef * polRef_exp).sum(-1)) - 1.0).max() < 1e-6
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        assert np.abs((polRef * samples["direction"]).sum(-1)).max() < 1e-6


def trackAngularEmission_pdf(x, n, cos_min=-1, cos_max=1, a=0.39, b=2.61, beta=1.0):
    """PDF of p(x) ~ exp(-bx^a)x^(a-1); x = |cos(theta) - cos(chev)|"""
    # shift x by cherenkov angle
    cos_chev = 1.0 / (beta * n)
    x = np.abs(x - cos_chev)

    # normalization constant
    int_lower = 1.0 - np.exp(-b * np.abs(cos_chev - cos_min) ** a)
    int_upper = 1.0 - np.exp(-b * np.abs(cos_chev - cos_max) ** a)
    # flip sign if cos_chev > cos_min/max
    int_lower *= np.sign(cos_min - cos_chev)
    int_upper *= np.sign(cos_max - cos_chev)
    norm = a * b / np.abs(int_upper - int_lower)

    # calculate pdf
    return np.exp(-b * x**a) * x ** (a - 1) * norm


@pytest.mark.parametrize(
    "n,cos_min,cos_max,a,b,beta",
    [
        [1.33, -1.0, 1.0, 0.39, 2.61, 1.0],
        [1.40, -1.0, -0.5, 0.41, 2.39, 0.9],
        [1.20, 0.8, 0.9, 0.44, 2.78, 1.0],
        [1.35, 0.1, 0.9, 0.41, 2.78, 0.7],
        [1.35, -0.1, 0.9, 0.41, 2.78, 1.0],
    ],
)
def test_trackAngularEmission_pdf(
    n: float,
    cos_min: float,
    cos_max: float,
    a: float,
    b: float,
    beta: float,
) -> None:
    # check wether the pdf integrates to 1
    est, err, *_ = integrate.quad(
        trackAngularEmission_pdf, cos_min, cos_max, (n, cos_min, cos_max, a, b, beta)
    )
    assert np.abs(est - 1.0) < 5e-4
    # check pdf(x) >= 0.0 for all x
    x = np.linspace(cos_min, cos_max, 500)
    assert np.all(trackAngularEmission_pdf(x, n, cos_min, cos_max, a, b, beta) >= 0.0)


@pytest.mark.parametrize(
    "n,a,b",
    [[1.33, 0.53, 3.3], [1.2, 0.86, 2.5], [1.45, 1.03, 3.1], [1.35, 1.13, 3.42]],
)
def test_particle_sampleEmissionAngle_full(n: float, a: float, b: float, shaderUtil):
    G = 1024 * 1024
    N = 32 * G
    N_BINS = 100

    tensor = hp.FloatTensor(N)
    buffer = hp.FloatBuffer(N)

    class Push(Structure):
        _fields_ = [("n", c_float), ("a", c_float), ("b", c_float)]

    philox = PhiloxRNG(key=0x01ABBA10)

    program = shaderUtil.createTestProgram(
        "lightsource.particle_sampleEmissionAngle.full.test.glsl",
        headers={"rng.glsl": philox.sourceCode},
    )
    program.bindParams(Samples=tensor)
    philox.bindParams(program, 0)
    philox.update(0)

    push = Push(n, a, b)
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), G))
        .Then(hp.retrieveTensor(tensor, buffer))
        .Submit()
        .wait()
    )

    # check result
    samples = buffer.numpy()
    hist, edges = np.histogram(samples, N_BINS, (-1.0, 1.0), density=True)
    pdf = trackAngularEmission_pdf(edges, n, a=a, b=b)
    exp_hist = 0.5 * (pdf[1:] + pdf[:-1])
    # testing the hist is a bit tricky as the pdf diverges at 1/n
    # -> ignore bin with this value and the ones left and right of it
    peak_bin = int((1.0 / n + 1) / (2.0 / N_BINS))
    err = np.abs(hist - exp_hist) / exp_hist
    assert err[: peak_bin - 1].max() < 0.05
    assert err[peak_bin + 2 :].max() < 0.05


@pytest.mark.parametrize(
    "n,a,b,cos_min,cos_max",
    [
        [1.33, 0.53, 3.3, -1.0, 1.0],
        [1.45, 0.86, 2.8, -0.5, 0.8],
        [1.35, 0.72, 3.1, -0.9, -0.2],
        [1.35, 1.03, 3.1, 0.8, 0.95],
    ],
)
def test_particle_sampleEmissionAngle_range(
    n: float, a: float, b: float, cos_min: float, cos_max: float, shaderUtil
) -> None:
    G = 1024 * 1024
    N = 32 * G
    N_BINS = 100

    sample_tensor = hp.FloatTensor(N)
    contrib_tensor = hp.FloatTensor(N)
    sample_buffer = hp.FloatBuffer(N)
    contrib_buffer = hp.FloatBuffer(N)

    class Push(Structure):
        _fields_ = [
            ("n", c_float),
            ("a", c_float),
            ("b", c_float),
            ("cos_min", c_float),
            ("cos_max", c_float),
        ]

    philox = PhiloxRNG(key=0x10500501)

    program = shaderUtil.createTestProgram(
        "lightsource.particle_sampleEmissionAngle.range.test.glsl",
        headers={"rng.glsl": philox.sourceCode},
    )
    program.bindParams(Values=sample_tensor, Contribs=contrib_tensor)
    philox.bindParams(program, 0)
    philox.update(0)

    push = Push(n, a, b, cos_min, cos_max)
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), G))
        .Then(hp.retrieveTensor(sample_tensor, sample_buffer))
        .And(hp.retrieveTensor(contrib_tensor, contrib_buffer))
        .Submit()
        .wait()
    )

    # check result
    samples = sample_buffer.numpy()
    hist, edges = np.histogram(samples, N_BINS, (cos_min, cos_max), density=True)
    pdf = trackAngularEmission_pdf(edges, n, cos_min, cos_max, a, b)
    exp_hist = 0.5 * (pdf[1:] + pdf[:-1])  # trapezoidal rule
    err = np.abs(hist - exp_hist) / exp_hist
    if cos_min <= 1.0 / n <= cos_max:
        # testing the hist is a bit tricky as the pdf diverges at 1/n
        # -> ignore bin with this value and the ones left and right of it
        cos_range = abs(cos_max - cos_min)
        peak_bin = int((1.0 / n - cos_min) / (cos_range / N_BINS))
        assert err[: peak_bin - 1].max() < 0.03
        assert err[peak_bin + 2 :].max() < 0.03
    else:
        assert err.max() < 0.01

    # we do not check contrib here, as we either have to specially deal with the
    # divergence at cos_chev or copy the code from the shader
    # Instead we check it implicitly in the light sources


@pytest.mark.parametrize("applyFrankTamm", [False, True])
def test_muonTrackLightSource_fwd(applyFrankTamm: bool) -> None:
    N = 256 * 1024
    lam_range = (300.0, 500.0) * u.nm
    E_muon = 10.0 * u.TeV
    startPos, startTime = np.array([1.0, 5.0, -2.0]) * u.m, 0.5 * u.us
    direction, dist = np.array([0.36, 0.48, 0.8]), 100.0 * u.m
    endPos, endTime = startPos + direction * dist, startTime + dist / u.c
    up = np.array([0.8, -0.6, 0.0])  # unit, orthogonal to direction

    # load water material
    water = WaterTestModel()
    store = MaterialStore([], media=[water.createMedium(*lam_range)])

    # create pipeline
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.UniformWavelengthSource(
        lambdaRange=lam_range, normalize=False
    )
    light = theia.light.MuonTrackLightSource(
        startPos, startTime, endPos, endTime, E_muon, applyFrankTamm=applyFrankTamm
    )
    sampler = theia.light.LightSampler(
        light, photons, N, rng=philox, medium=store.media["water"]
    )
    # run
    runPipeline(sampler.collectStages())
    result = sampler.lightQueue.view(0)
    photons = sampler.wavelengthQueue.view(0)

    # check all sampled postions are on the track
    distStart = np.sqrt(np.square(startPos - result["position"]).sum(-1))
    distEnd = np.sqrt(np.square(endPos - result["position"]).sum(-1))
    assert np.allclose(distStart + distEnd, dist)
    expTime = startTime + distStart / u.c
    assert np.allclose(result["startTime"], expTime)
    # check we use the whole track
    assert distStart.min() < 0.05
    assert distEnd.min() < 0.05
    # check direction
    assert np.allclose(np.square(result["direction"]).sum(-1), 1.0)
    cos_theta = np.multiply(result["direction"], direction).sum(-1)
    # since the angular emission pdf is conditional on the refractive index
    # a direct test is a bit complicated (we cannot just create a histogram or
    # use a KS test). For now we just check the range and trust the check of
    # the MC estimate to be enough
    assert cos_theta.max() > 0.999
    assert cos_theta.min() < -0.999
    # check phi is uniform
    cos_phi = np.multiply(result["direction"], up).sum(-1)
    sin_phi = np.multiply(np.cross(result["direction"], up), direction).sum(-1)
    phi = np.arctan2(sin_phi, cos_phi)
    phi = (phi + np.pi) / (2.0 * np.pi)
    ks = stats.kstest(phi[::100], "uniform")
    assert ks.pvalue > 0.05
    # check MC estimate converges to correct result, which is the total amount
    # of cherenkov photons produced
    energyScale = light.getParam("_energyScale")
    if applyFrankTamm:
        lam = np.linspace(*lam_range, 1025)
        dlam = (lam_range[1] - lam_range[0]) / (len(lam) - 1)
        n = water.refractive_index(lam)
        ft = theia.light.frankTamm(lam, n)
        dN_dx = integrate.romb(ft, dlam)
        nPhotons = dN_dx * dist * energyScale
        contrib = result["contrib"] * photons["contrib"]
        assert np.abs(1.0 - contrib.mean() / nPhotons) < 5e-4
    else:
        # if we do not apply the frank tamm formula this reduces to simply
        # sampling a point along the track, scaled by the energy scale
        assert np.allclose(result["contrib"], dist * energyScale)


@pytest.mark.parametrize(
    "observer,applyFrankTamm",
    [
        [(15.0, 30.0, 60.0) * u.m, True],
        [(-10.0, -30.0, 0.0) * u.m, False],
        [(80.0, 80.0, 80.0) * u.m, True],
    ],
)
def test_muonTrackLightSource_bwd(
    observer: tuple[float, float, float], applyFrankTamm: bool
) -> None:
    N = 256 * 1024
    lam_range = (300.0, 500.0) * u.nm
    E_muon = 10.0 * u.TeV
    startPos, startTime = np.array([1.0, 5.0, -2.0]) * u.m, 0.5 * u.us
    direction, dist = np.array([0.36, 0.48, 0.8]), 100.0 * u.m
    endPos, endTime = startPos + direction * dist, startTime + dist / u.c

    # load water material
    water = WaterTestModel()
    store = MaterialStore([], media=[water.createMedium(*lam_range)])

    # create pipeline
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.UniformWavelengthSource(lambdaRange=lam_range)
    light = theia.light.MuonTrackLightSource(
        startPos, startTime, endPos, endTime, E_muon, applyFrankTamm=applyFrankTamm
    )
    sampler = BackwardLightSampler(
        N, light, photons, rng=philox, observer=observer, medium=store.media["water"]
    )
    # run
    runPipeline(sampler.collectStages())
    result = sampler.getResults(0)

    # check all sampled postions are on the track
    distStart = np.sqrt(np.square(startPos - result["position"]).sum(-1))
    distEnd = np.sqrt(np.square(endPos - result["position"]).sum(-1))
    assert np.allclose(distStart + distEnd, dist)
    expTime = startTime + distStart / u.c
    assert np.allclose(result["startTime"], expTime)
    # check we use the whole track
    assert distStart.min() < 0.05
    assert distEnd.min() < 0.05
    # check we aim at observer
    assert np.allclose(result["observer"], observer)
    expDir = observer - result["position"]
    expDir /= np.sqrt(np.square(expDir).sum(-1))[:, None]
    assert np.allclose(result["direction"], expDir)

    # instead of checking the individual sample contributions, we check the
    # corresponding MC estimate, which is the expected number of photons
    # arriving at the observer.
    energyScale = light.getParam("_energyScale")
    a_angular = light.getParam("_a_angular")
    b_angular = light.getParam("_b_angular")

    # integrand
    def f(lam, x):
        # evaluate frank tamm formula
        n = water.refractive_index(lam)
        dN_dx = theia.light.frankTamm(lam, n) if applyFrankTamm else 1.0
        # calculate emission angle
        p = startPos + x * direction
        dir_p = observer - p
        r = np.sqrt(np.square(dir_p).sum(-1))
        dir_p /= r
        cos_theta = np.multiply(dir_p, direction).sum()
        # evaluate emission profile
        ang = trackAngularEmission_pdf(cos_theta, n, a=a_angular, b=b_angular)
        area = 4.0 * np.pi * r**2

        return dN_dx * energyScale * ang / area

    est, err = integrate.dblquad(f, 0, dist, *lam_range, epsrel=1e-4)
    est /= lam_range[1] - lam_range[0]
    assert np.abs(1.0 - result["contrib"].mean() / est) < 5e-3


@pytest.mark.parametrize("applyFrankTamm", [False, True])
@pytest.mark.parametrize("particle", ["e-", "pi+"])
def test_particleCascadeLightSource_fwd(applyFrankTamm: bool, particle: str) -> None:
    N = 256 * 1024
    lam_range = (300.0, 500.0) * u.nm
    startPos, startTime = np.array([1.0, 5.0, -2.0]) * u.m, 0.5 * u.us
    direction = np.array([0.36, 0.48, 0.8])
    up = np.array([0.8, -0.6, 0.0])  # unit, orthogonal to direction
    E_primary = 1.0 * u.TeV
    if particle == "e-":
        primary = theia.cascades.ParticleType.E_MINUS
    else:
        primary = theia.cascades.ParticleType.PI_PLUS
    p = theia.cascades.Particle(primary, startPos, direction, E_primary, startTime)
    params = theia.cascades.createParamsFromParticle(p, lightSourceName="")[1]

    # load water material
    water = WaterTestModel()
    store = MaterialStore([], media=[water.createMedium(*lam_range)])

    # create pipeline
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.UniformWavelengthSource(lambdaRange=lam_range)
    light = theia.light.ParticleCascadeLightSource(
        **params, applyFrankTamm=applyFrankTamm
    )
    sampler = theia.light.LightSampler(
        light, photons, N, rng=philox, medium=store.media["water"]
    )
    # run
    runPipeline(sampler.collectStages())
    result = sampler.lightQueue.view(0)
    photons = sampler.wavelengthQueue.view(0)

    # check all sampled positions are on the track
    dirPos = result["position"] - startPos
    distPos = np.sqrt(np.square(dirPos).sum(-1))
    dirPos = dirPos / distPos[:, None]
    assert np.allclose(dirPos[distPos > 0.0], direction)
    expTime = startTime + distPos / u.c
    assert np.allclose(expTime, result["startTime"])
    # check we sample along the whole track
    z_max = (params["a_long"] - 1) * params["b_long"]  # mode of gamma dist
    assert distPos.min() < 0.1 * z_max
    assert distPos.max() > 4.0 * z_max  # in theory infinite, but really unlikely
    # check direction
    assert np.allclose(np.square(result["direction"]).sum(-1), 1.0)
    cos_theta = np.multiply(result["direction"], direction).sum(-1)
    # since the angular emission pdf is conditional on the refractive index
    # a direct test is a bit complicated (we cannot just create a histogram or
    # use a KS test). For now we just check the range and trust the check of
    # the MC estimate to be enough
    assert cos_theta.min() < -0.999
    assert cos_theta.max() > 0.999
    # check phi is uniform
    cos_phi = np.multiply(result["direction"], up).sum(-1)
    sin_phi = np.multiply(np.cross(result["direction"], up), direction).sum(-1)
    phi = np.arctan2(sin_phi, cos_phi)
    phi = (phi + np.pi) / (2.0 * np.pi)
    ks = stats.kstest(phi[::100], "uniform")
    assert ks.pvalue > 0.05

    # instead of checking the individual sample contributions we check the MC
    # estimate which is the total number of photons produced
    est = result["contrib"].mean()
    expEst = params["effectiveLength"]
    if applyFrankTamm:
        lam = np.linspace(*lam_range, 1025)
        dlam = (lam_range[1] - lam_range[0]) / (len(lam) - 1)
        n = water.refractive_index(lam)
        dN_dx = integrate.romb(theia.light.frankTamm(lam, n), dlam)
        expEst = dN_dx * params["effectiveLength"]
        # we miss the contrib from the wavelength source in the sample
        expEst /= lam_range[1] - lam_range[0]
    assert np.abs(1.0 - est / expEst) < 5e-4


@pytest.mark.parametrize(
    "observer,applyFrankTamm",
    [
        [(5.0, 3.0, 6.0) * u.m, True],
        [(-1.0, -3.0, 0.0) * u.m, False],
        [(20.0, 20.0, 20.0) * u.m, True],
    ],
)
@pytest.mark.parametrize("particle", ["e-", "pi+"])
def test_particleCascadeLightSource_bwd(
    observer: tuple[float, float, float], particle: str, applyFrankTamm: bool
) -> None:
    N = 1024 * 1024
    lam_range = (300.0, 500.0) * u.nm
    startPos, startTime = np.array([1.0, 5.0, -2.0]) * u.m, 0.5 * u.us
    direction = np.array([0.36, 0.48, 0.8])
    E_primary = 1.0 * u.TeV
    if particle == "e-":
        primary = theia.cascades.ParticleType.E_MINUS
    else:
        primary = theia.cascades.ParticleType.PI_PLUS
    p = theia.cascades.Particle(primary, startPos, direction, E_primary, startTime)
    params = theia.cascades.createParamsFromParticle(p, lightSourceName="")[1]
    a_long, b_long = params["a_long"], params["b_long"]

    # load water material
    water = WaterTestModel()
    store = MaterialStore([], media=[water.createMedium(*lam_range)])

    # create pipeline
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.UniformWavelengthSource(lambdaRange=lam_range)
    light = theia.light.ParticleCascadeLightSource(
        **params, applyFrankTamm=applyFrankTamm
    )
    sampler = BackwardLightSampler(
        N, light, photons, rng=philox, observer=observer, medium=store.media["water"]
    )
    # run
    runPipeline(sampler.collectStages())
    result = sampler.getResults(0)

    # check all sampled positions are on the track
    dirPos = result["position"] - startPos
    distPos = np.sqrt(np.square(dirPos).sum(-1))
    dirPos /= distPos[:, None]
    assert np.allclose(dirPos[distPos > 0.0], direction)
    expTime = startTime + distPos / u.c
    assert np.allclose(expTime, result["startTime"])
    # check we sample along the whole track
    z_max = (a_long - 1) * b_long  # mode of gamma dist
    assert distPos.min() < 0.1 * z_max
    assert distPos.max() > 4.0 * z_max  # in theory infinite, but really unlikely
    # check we aim at the observer
    assert np.allclose(result["observer"], observer)
    expDir = observer - result["position"]
    expDir /= np.sqrt(np.square(expDir).sum(-1))[:, None]
    assert np.allclose(result["direction"], expDir)

    # instead of checking the individual sample contributions, we check the
    # corresponding MC estimate, which is the expected number of photons
    # arriving at the observer.
    gamma = stats.gamma(a_long)

    # integrand
    def f(lam, x):
        # evaluate frank tamm formula
        n = water.refractive_index(lam)
        result = theia.light.frankTamm(lam, n) if applyFrankTamm else 1.0
        # calculate emission position
        result *= gamma.pdf(x / b_long) / b_long
        # calculate emission angle
        p = startPos + x * direction
        dir_p = observer - p
        r = np.sqrt(np.square(dir_p).sum(-1))
        dir_p /= r
        cos_theta = np.multiply(dir_p, direction).sum()
        # evaluate emission profile
        a, b = params["a_angular"], params["b_angular"]
        result *= trackAngularEmission_pdf(cos_theta, n, a=a, b=b)
        result *= params["effectiveLength"]
        result /= 4.0 * np.pi * (r**2)
        # done
        return result

    est, err = integrate.dblquad(f, 0, 10.0 * z_max, *lam_range, epsrel=1e-3)
    # we are missing the lambda contrib in our sampler -> get rid of it
    est /= lam_range[1] - lam_range[0]
    assert np.abs(1.0 - result["contrib"].mean() / est) < 0.07
