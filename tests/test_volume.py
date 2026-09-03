import pytest

import hephaistos.pipeline as pl
import numpy as np

from theia.camera import SphereCamera
from theia.light import (
    ConstWavelengthSource,
    SphericalLightSource,
    UniformWavelengthSource,
)
from theia.material import MaterialStore
from theia.model import DispersionFreeMedium, HenyeyGreensteinPhaseFunction
from theia.model import (
    optical_constant,
    optical_property,
    numerical_quantile_property,
    numerical_sampler_property,
)
from theia.random import PhiloxRNG
from theia.ray import UnpolarizedRay
from theia.testing import VolumeInteractionSampler, VolumeScatteringSampler
from theia.trace import EventResultCode
import theia.units as u
import theia.volume


class MediumModel(DispersionFreeMedium, HenyeyGreensteinPhaseFunction):
    def __init__(
        self, mu_a: float, mu_s: float, g: float, *, n: float = 1.33, ng: float = 1.33
    ) -> None:
        super().__init__(n=n, ng=ng, mu_a=mu_a, mu_s=mu_s, g=g, name="homogenous")


@pytest.mark.parametrize("absorb", [True, False])
@pytest.mark.parametrize(
    "particle,camera,sampleIntLength",
    [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
    ],
)
def test_Attenuating(particle: bool, camera: bool, sampleIntLength: bool, absorb: bool):
    N = 32 * 1024
    lam = 600.0 * u.nm
    mu_a, mu_s = 0.02 / u.m, 0.08 / u.m
    g = 0.7

    # create medium
    volModel = theia.volume.Attenuating(absorb=absorb)
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium(physicModel=volModel)
    matStore = MaterialStore([], media=[medium])
    medIdx = matStore.media["homogenous"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle, transient=False)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = SphereCamera(mediumIdx=medIdx)
    else:
        source = SphericalLightSource(photons, emitParticles=particle, mediumIdx=medIdx)
        photons = None
    sampler = VolumeInteractionSampler(
        N,
        ray,
        volModel,
        matStore,
        rng,
        source,
        photons,
        sampleCoef=float("NaN") if sampleIntLength else 0.05 / u.m,
        propagateRay=True,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    rayItem = ray.backwardItem if camera else ray.forwardItem
    assert rayItem is not None
    for name, _ in rayItem._fields_:
        if name in {"direction", "contrib", "position", "time"}:
            continue
        assert np.allclose(result[name + "In"], result[name + "Out"])
    hitMask = result["hit"] == 1
    absorbed = result["resultCode"] == EventResultCode.RAY_ABSORBED
    if particle:
        absorb = True
    if absorb:
        assert absorbed.sum() > 0
    else:
        assert absorbed.sum() == 0
    mask = ~absorbed & ~hitMask
    assert np.all(result["resultCode"][mask] == EventResultCode.RAY_SCATTERED)
    if not particle:
        contrib = result["contribOut"] / result["contribIn"]
        exp_contrib = np.ones_like(contrib)
        if not absorb:
            exp_contrib[~hitMask] *= mu_s / (mu_s + mu_a)
        if not sampleIntLength:
            d = np.sqrt(np.square(result["positionIn"] - result["positionOut"]).sum(-1))
            exp_contrib *= np.exp(-(mu_s + mu_a) * d)
            exp_contrib[~hitMask] *= mu_s + mu_a
        assert np.allclose(contrib[~absorbed], exp_contrib[~absorbed])
    # TODO: maybe add test to check MC estimate?
    # A more thorough test would be nice, but that'll do for now
    cos_theta = np.multiply(result["directionIn"], result["directionOut"]).sum(-1)
    assert np.abs(cos_theta.mean() - g) < 0.1  # large error for small statistic


@pytest.mark.parametrize("camera", [True, False])
def test_Attenuating_scatter(camera: bool):
    N = 32 * 1024
    lam = 600.0 * u.nm
    mu_a, mu_s = 0.02 / u.m, 0.08 / u.m
    g = 0.7

    # create medium
    volModel = theia.volume.Attenuating()
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium(physicModel=volModel)
    matStore = MaterialStore([], media=[medium])
    medIdx = matStore.media["homogenous"]

    # create pipeline
    ray = UnpolarizedRay(transient=False)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = SphereCamera(mediumIdx=medIdx)
    else:
        source = SphericalLightSource(photons, mediumIdx=medIdx)
        photons = None
    sampler = VolumeScatteringSampler(
        N,
        ray,
        volModel,
        matStore,
        rng,
        source,
        photons,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    rayItem = ray.backwardItem if camera else ray.forwardItem
    assert rayItem is not None
    for name, _ in rayItem._fields_:
        if name in {"direction", "contrib", "time"}:
            continue
        assert np.allclose(result[name + "In"], result[name + "Out"])
    assert np.all(result["scatterResult"] == EventResultCode.SUCCESS)
    ct_sampled = np.multiply(result["directionIn"], result["sampleDir"]).sum(-1)
    ct_rand = np.multiply(result["directionIn"], result["randomDir"]).sum(-1)
    assert np.allclose(result["directionOut"], result["randomDir"])
    assert np.allclose(result["probSampled"], result["probEval"], rtol=5e-5)
    prob_sampled = np.exp(model.log_phase_function(ct_sampled))
    assert np.allclose(result["probEval"], prob_sampled, atol=5e-4)
    prob_rand = np.exp(model.log_phase_function(ct_rand))
    assert np.allclose(result["probRandom"], prob_rand, atol=5e-4)
    contrib = result["contribOut"] / result["contribIn"]
    assert np.allclose(contrib, mu_s * prob_rand, atol=5e-5)


@pytest.mark.parametrize(
    "particle,camera,sampleIntLength",
    [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
    ],
)
def test_Transparent(particle: bool, camera: bool, sampleIntLength: bool):
    N = 32 * 1024
    lam = 600.0 * u.nm

    # create medium
    volModel = theia.volume.Transparent()
    medium = MediumModel(0.0, 0.0, 0.0).createMedium(physicModel=volModel)
    matStore = MaterialStore([], media=[medium])
    medIdx = matStore.media["homogenous"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = SphereCamera(mediumIdx=medIdx)
    else:
        source = SphericalLightSource(photons, mediumIdx=medIdx, emitParticles=particle)
        photons = None
    sampler = VolumeInteractionSampler(
        N,
        ray,
        volModel,
        matStore,
        rng,
        source,
        photons,
        sampleCoef=float("NaN") if sampleIntLength else 0.01 / u.m,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    rayItem = ray.backwardItem if camera else ray.forwardItem
    assert rayItem is not None
    for name, _ in rayItem._fields_:
        assert np.allclose(result[name + "In"], result[name + "Out"])
    assert np.all(result["resultCode"] == EventResultCode.SUCCESS)


def _sinWindowFn(x, x_min, x_max):
    x = np.asarray(x)
    a = x_max - x_min
    mask = (x >= x_min) & (x <= x_max)
    return 0.5 * np.pi / a * np.sin(np.pi / a * (x - x_min)) * mask


class FluorescentTestMediumModel(DispersionFreeMedium, HenyeyGreensteinPhaseFunction):
    def __init__(
        self, mu_a: float, mu_s: float, mu_f: float, g: float, qe: float, delta_t: float
    ) -> None:
        super().__init__(
            n=1.33,
            mu_a=mu_a,
            mu_s=mu_s,
            g=g,
            name="fluorescent",
        )
        self._mu_f = mu_f
        self._qe = qe
        self._delta_t = delta_t
        self.restrictWavelengthRange((100.0, 800.0) * u.nm)

    @optical_constant
    def fluorescence_efficiency(self) -> float:
        return self._qe

    @optical_constant
    def fluorescence_time_shift(self) -> float:
        return self._delta_t

    @optical_property
    def fluorescence_coef(self, wavelength):
        x = u.convert(wavelength, u.nm)
        return _sinWindowFn(x, 200.0, 350.0) * self._mu_f

    @optical_property(range=(300.0, 500.0) * u.nm)
    def fluorescence_emission(self, wavelength):
        x = u.convert(wavelength, u.nm)
        return _sinWindowFn(x, 300.0, 500.0)

    fluorescence_emission_quantile = numerical_quantile_property(fluorescence_emission)
    fluorescence_emission_sampling = numerical_sampler_property(fluorescence_emission)


@pytest.mark.parametrize("upshift", [True, False])
@pytest.mark.parametrize("absorb", [True, False])
@pytest.mark.parametrize(
    "particle,sampleIntLength",
    [
        (False, True),
        (False, False),
        (True, True),
    ],
)
def test_Fluorescent(
    particle: bool, sampleIntLength: bool, absorb: bool, upshift: bool
):
    N = 32 * 1024
    lam_range = (150.0, 600.0) * u.nm
    mu_a = 0.02 / u.m
    mu_s = 0.06 / u.m
    mu_f = 1.0 / u.m  # and nm^-1
    g = 0.7
    qe = 0.8
    delta_t = 25.0 * u.ns

    # create medium
    volModel = theia.volume.Fluorescent(absorb=absorb, allowWavelengthUpShift=upshift)
    model = FluorescentTestMediumModel(mu_a, mu_s, mu_f, g, qe, delta_t)
    medium = model.createMedium(physicModel=volModel)
    matStore = MaterialStore([], media=[medium])
    medIdx = matStore.media["fluorescent"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = UniformWavelengthSource(lambdaRange=lam_range)
    source = SphericalLightSource(photons, emitParticles=particle, mediumIdx=medIdx)
    sampler = VolumeInteractionSampler(
        N,
        ray,
        volModel,
        matStore,
        rng,
        source,
        sampleCoef=float("NaN") if sampleIntLength else 0.05 / u.m,
        propagateRay=True,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    for name, _ in ray.forwardItem._fields_:
        if name in {"position", "direction", "contrib", "time", "wavelength"}:
            continue
        assert np.allclose(result[name + "In"], result[name + "Out"])
    hitMask = result["hit"] == 1
    absorbed = result["resultCode"] == EventResultCode.RAY_ABSORBED
    fluorMask = result["wavelengthIn"] != result["wavelengthOut"]
    assert fluorMask.sum() > 0
    dt = result["timeOut"] - result["timeIn"]
    d = np.sqrt(np.square(result["positionIn"] - result["positionOut"]).sum(-1))
    vg = model.group_velocity(500.0)  # const
    dt -= d / vg  # remove propagation time
    assert np.allclose(dt[fluorMask], delta_t)
    assert np.allclose(dt[~fluorMask], 0.0, atol=5e-4)
    if upshift:
        assert (result["wavelengthIn"] > result["wavelengthOut"]).sum() > 0
    else:
        assert (result["wavelengthIn"] > result["wavelengthOut"]).sum() == 0
    if particle:
        absorb = True
    if absorb:
        assert absorbed.sum() > 0
    else:
        assert absorbed.sum() == 0
    mask = ~absorbed & ~hitMask
    assert np.all(result["resultCode"][mask] == EventResultCode.RAY_SCATTERED)
    if not particle:
        contrib = result["contribOut"] / result["contribIn"]
        exp_contrib = np.ones_like(contrib)
        mu_f_r = model.fluorescence_coef(result["wavelengthIn"])
        mu_e = mu_f_r + mu_a + mu_s
        mu_f_r *= qe
        if not absorb:
            exp_contrib[~hitMask] *= ((mu_s + mu_f_r) / mu_e)[~hitMask]
        if not sampleIntLength:

            exp_contrib *= np.exp(-mu_e * d)
            exp_contrib[~hitMask] *= mu_e[~hitMask]
        assert np.allclose(contrib[~absorbed], exp_contrib[~absorbed], atol=5e-4)
    # TODO: maybe add a test checking the scattering angles?


@pytest.mark.parametrize("upshift", [True, False])
def test_Fluorescent_scattering(upshift: bool):
    N = 32 * 1024
    lam_range = (150.0, 400.0) * u.nm
    mu_a = 0.02 / u.m
    mu_s = 0.03 / u.m
    mu_f = 1.0 / u.m  # and nm^-1
    g = 0.7
    qe = 0.8
    delta_t = 25.0 * u.ns

    # create medium
    volModel = theia.volume.Fluorescent(allowWavelengthUpShift=upshift)
    model = FluorescentTestMediumModel(mu_a, mu_s, mu_f, g, qe, delta_t)
    medium = model.createMedium(physicModel=volModel)
    matStore = MaterialStore([], media=[medium])
    medIdx = matStore.media["fluorescent"]

    # create pipeline
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xABBA)
    photons = UniformWavelengthSource(lambdaRange=lam_range)
    source = SphericalLightSource(photons, mediumIdx=medIdx)
    sampler = VolumeScatteringSampler(
        N,
        ray,
        volModel,
        matStore,
        rng,
        source,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    for name, _ in ray.forwardItem._fields_:
        if name in {"direction", "contrib", "time", "wavelength"}:
            continue
        assert np.allclose(result[name + "In"], result[name + "Out"])
    assert np.all(result["scatterResult"] == EventResultCode.SUCCESS)
    f_mask = result["wavelengthIn"] != result["wavelengthOut"]
    assert f_mask.sum() != 0
    dt = result["timeOut"] - result["timeIn"]
    assert np.allclose(dt[f_mask], delta_t)
    assert np.allclose(dt[~f_mask], 0.0)
    if upshift:
        assert (result["wavelengthIn"] > result["wavelengthOut"]).sum() > 0
    else:
        assert (result["wavelengthIn"] > result["wavelengthOut"]).sum() == 0
    ct_sampled = np.multiply(result["directionIn"], result["sampleDir"]).sum(-1)
    ct_rand = np.multiply(result["directionIn"], result["randomDir"]).sum(-1)
    assert np.allclose(result["directionOut"], result["randomDir"])
    assert np.allclose(result["probSampled"], result["probEval"])
    mu_f_r = model.fluorescence_coef(result["wavelengthIn"]) * qe
    p_scatter = mu_s / (mu_s + mu_f_r) * np.exp(model.log_phase_function(ct_sampled))
    prob_sampled = p_scatter + mu_f_r / (mu_s + mu_f_r) / (4.0 * np.pi)
    assert np.allclose(result["probEval"], prob_sampled, atol=2e-3)
    p_scatter_r = mu_s / (mu_s + mu_f_r) * np.exp(model.log_phase_function(ct_rand))
    prob_rand = p_scatter_r + mu_f_r / (mu_s + mu_f_r) / (4.0 * np.pi)
    assert np.allclose(result["probRandom"], prob_rand, atol=5e-4)
    contrib = result["contribOut"] / result["contribIn"]
    contrib_exp = np.ones_like(contrib)
    contrib_exp *= mu_s + mu_f_r
    contrib_exp[f_mask] *= 1.0 / (4.0 * np.pi)
    contrib_exp[~f_mask] *= np.exp(model.log_phase_function(ct_rand[~f_mask]))
    assert np.allclose(contrib, contrib_exp, atol=5e-5)


def test_Fluorescent_exponentialDecay():
    # with timeModel="exponential" the fluorescence re-emission delay is drawn
    # from an exponential decay with time constant fluorescence_time
    N = 256 * 1024  # enough fluorescent scatters for the histogram
    lam_range = (150.0, 600.0) * u.nm
    mu_a = 0.02 / u.m
    mu_s = 0.06 / u.m
    mu_f = 1.0 / u.m  # and nm^-1
    g = 0.7
    qe = 0.8
    delta_t = 25.0 * u.ns

    # create medium
    volModel = theia.volume.Fluorescent(timeModel="exponential")
    model = FluorescentTestMediumModel(mu_a, mu_s, mu_f, g, qe, delta_t)
    medium = model.createMedium(physicModel=volModel)
    matStore = MaterialStore([], media=[medium])
    medIdx = matStore.media["fluorescent"]

    # create pipeline
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xABBA)
    photons = UniformWavelengthSource(lambdaRange=lam_range)
    source = SphericalLightSource(photons, mediumIdx=medIdx)
    sampler = VolumeInteractionSampler(
        N,
        ray,
        volModel,
        matStore,
        rng,
        source,
        sampleCoef=0.05 / u.m,
        propagateRay=True,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    fluorMask = result["wavelengthIn"] != result["wavelengthOut"]
    assert fluorMask.sum() > 0
    dt = result["timeOut"] - result["timeIn"]
    d = np.sqrt(np.square(result["positionIn"] - result["positionOut"]).sum(-1))
    vg = model.group_velocity(500.0)  # constant
    dt = (dt - d / vg)[fluorMask]

    # an exponential distribution has mean and standard deviation equal to its
    # time constant
    n = len(dt)
    assert dt.min() > -1e-5
    assert abs(np.mean(dt) - delta_t) < 3.0 * delta_t / np.sqrt(n)
    assert abs(np.std(dt) - delta_t) < 3.0 * delta_t / np.sqrt(n)
    # check distribution via histogram
    hist, edges = np.histogram(dt, bins=40, range=(0.0, 6.0 * delta_t))
    hist = hist / n
    exp_hist = np.diff(1.0 - np.exp(-edges / delta_t))
    sigma = np.sqrt(exp_hist * (1.0 - exp_hist) / n)
    assert np.all(np.abs(hist - exp_hist) < 3.0 * sigma + 1e-5)
