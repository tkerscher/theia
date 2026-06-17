import pytest

import hephaistos.pipeline as pl
import numpy as np

from theia.camera import SphereCamera
from theia.light import ConstWavelengthSource, SphericalLightSource
from theia.material import MaterialStore
from theia.model import DispersionFreeMedium, HenyeyGreensteinPhaseFunction
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
