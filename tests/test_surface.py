import pytest

import hephaistos.pipeline as pl
import numpy as np

from theia.camera import ConeCamera
from theia.light import ConeLightSource, ConstWavelengthSource
from theia.material import Material, MaterialStore, Medium, VACUUM_IDX
from theia.model import BK7Model, PureWaterModel
from theia.property import FloatProperty, Property, TableProperty
from theia.random import PhiloxRNG
from theia.ray import UnpolarizedRay
from theia.testing import SurfaceInteractionSampler, SurfaceScatteringSampler
from theia.trace import EventResultCode
import theia.units as u
import theia.surface


def reflect(i, n):
    ct = np.multiply(i, n[None, :]).sum(1)
    return i - 2.0 * ct[:, None] * n


def refract(i, n, ni, no):
    eta = ni / no
    ct = np.multiply(i, n[None, :]).sum(1)
    k = 1.0 - eta * eta * (1.0 - ct * ct)
    return eta[:, None] * i - (eta * ct + np.sqrt(k))[:, None] * n


def reflectance(i, n, ni, no):
    ci = np.abs(np.multiply(i, n[None, :]).sum(1))
    si = np.sqrt(np.clip(1.0 - np.square(ci), 0.0, 1.0))
    so = si * ni / no
    co = np.sqrt(np.clip(1.0 - np.square(so), 0.0, 1.0))
    rs = (ni * ci - no * co) / (ni * ci + no * co)
    rp = (no * ci - ni * co) / (no * ci + ni * co)
    return 0.5 * (rs * rs + rp * rp)


def reflectance_metal(i, n, ni, no, k):
    no = no + 1.0j * k
    ci = np.abs(np.multiply(i, n[None, :]).sum(1))
    si = np.sqrt(np.clip(1.0 - np.square(ci), 0.0, 1.0))
    co = np.sqrt(1.0 - (ni / no * si) ** 2, dtype=complex)
    rs = (ni * ci - no * co) / (ni * ci + no * co)
    rp = (ni * co - no * ci) / (ni * co + no * ci)
    return 0.5 * (rs.real**2 + rs.imag**2 + rp.real**2 + rp.imag**2)


@pytest.mark.parametrize(
    "particle,camera,flags",
    [
        (True, False, "TR"),
        (True, False, "T"),
        (True, False, "R"),
        (True, False, "DR"),
        (False, True, "TR"),
        (False, True, "T"),
        (False, True, "R"),
        (False, False, "TR"),
        (False, False, "T"),
        (False, False, "R"),
        (False, False, "DR"),
    ],
)
def test_DielectricSurface(particle: bool, camera: bool, flags: str):
    N = 32 * 1024
    lam = 600.0 * u.nm
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    surface = theia.surface.DielectricSurface()
    mat = Material("mat", None, water, surface, flags=flags)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = ConeCamera(
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            objectId=objectId,
        )
    else:
        source = ConeLightSource(
            photons,
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
            emitParticles=particle,
        )
        photons = None
    sampler = SurfaceInteractionSampler(
        N,
        ray,
        surface,
        matStore,
        rng,
        source,
        photons,
        material="mat",
        surfaceNormal=normal,
        objectId=objectId,
        sampleTargetHit=not camera,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)
    if not camera:
        # only consider valid hits
        valid = result["hitSuccess"] == 1
        assert np.all(result["wavelengthHit"][valid] == lam)
        assert np.all(result["objectIdHit"][valid] == objectId)
    if flags == "D":
        # neither reflect nor transmit flag -> absorb
        assert np.all(result["hitResult"] == EventResultCode.RAY_ABSORBED)
        return
    absorbed = result["hitResult"] == EventResultCode.RAY_ABSORBED
    assert np.all(result["hitResult"][~absorbed] == EventResultCode.RAY_HIT)

    normal = np.array(normal)
    cosNrm = np.multiply(result["directionOut"], normal[None, :]).sum(1)
    trans = (cosNrm < 0.0) & ~absorbed
    refl = (cosNrm >= 0.0) & ~absorbed
    ni = waterModel.refractive_index(lam) * np.ones(N)
    no = np.ones(N)
    t = refract(result["directionIn"], normal, ni, no)
    r = reflect(result["directionIn"], normal)
    R = reflectance(result["directionIn"], normal, ni, no)
    if not particle:
        c = result["contribIn"]
    if camera:
        # in backward mode transmittance scales with a factor eta^2
        eta = ni / no
        c = np.copy(result["contribIn"])
        c[trans] = (c * eta * eta)[trans]

    # check we prevent self intersection
    cosPos = np.multiply(result["positionOut"], normal[None, :]).sum(-1)
    assert np.all(cosPos[trans] < 0.0)
    assert np.all(cosPos[refl] > 0.0)

    assert np.allclose(result["directionOut"][trans], t[trans], atol=1e-5)
    assert np.allclose(result["directionOut"][refl], r[refl], atol=1e-5)
    assert np.all(result["mediumIdxOut"][trans] == VACUUM_IDX)
    assert np.all(result["mediumIdxOut"][refl] == waterIdx)

    if flags == "R" or flags == "DR":
        assert trans.sum() == 0
        if particle:
            assert absorbed.sum() > 0
        else:
            cO = result["contribOut"]
            assert np.allclose(cO[~absorbed], (c * R)[~absorbed], atol=5e-4)
    if flags == "T":
        assert refl.sum() == 0
        assert absorbed.sum() > 0  # total internal reflection
        if not particle:
            cO = result["contribOut"]
            assert np.allclose(cO[~absorbed], (c * (1.0 - R))[~absorbed], atol=5e-4)
        if not particle and not camera:
            cH = result["contribHit"]
            assert np.allclose(cH[~absorbed], (c * (1.0 - R))[~absorbed], atol=5e-4)
    if flags == "TR":
        assert refl.sum() > 0
        assert trans.sum() > 0
        assert absorbed.sum() == 0
        if particle:
            # check particle is not both reflected and detected
            np.all((result["hitSuccess"] == 0) == refl)
        else:
            assert np.allclose(result["contribOut"], c)
        if not particle and not camera:
            cH = result["contribHit"]
            assert np.allclose(cH[~absorbed], (c * (1.0 - R))[~absorbed], atol=5e-5)


def createConstMetal(name: str, n: float, k: float) -> Medium:
    """creates medium with constant complex refractive index"""
    props: dict[str, Property] = {
        "refractive_index": TableProperty.createConstTable(n),
        "imag_refractive_index": TableProperty.createConstTable(k),
    }
    return Medium(name, (100.0, 1000.0) * u.nm, props)


@pytest.mark.parametrize("absorb", [True, False])
@pytest.mark.parametrize("R", [-1.0, 0.0, 0.65, 1.0])
@pytest.mark.parametrize(
    "particle,camera,flags",
    [
        (True, False, "TR"),
        (True, False, "T"),
        (True, False, "R"),
        (True, False, "DR"),
        (False, True, "TR"),
        (False, True, "T"),
        (False, True, "R"),
        (False, False, "TR"),
        (False, False, "T"),
        (False, False, "R"),
        (False, False, "DR"),
    ],
)
def test_MetallicSurface(
    particle: bool, camera: bool, flags: str, R: float, absorb: bool
):
    N = 32 * 1024
    lam = 600.0 * u.nm
    n, k = 1.9, 2.35
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    metal = createConstMetal("metal", n, k)
    props: dict[str, Property] = {"reflectivity": TableProperty.createConstTable(R)}
    surface = theia.surface.MetallicSurface(absorb=absorb)
    mat = Material("mat", metal, water, surface, flags=flags, properties=props)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = ConeCamera(
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            objectId=objectId,
        )
    else:
        source = ConeLightSource(
            photons,
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
            emitParticles=particle,
        )
        photons = None
    sampler = SurfaceInteractionSampler(
        N,
        ray,
        surface,
        matStore,
        rng,
        source,
        photons,
        material="mat",
        surfaceNormal=normal,
        objectId=objectId,
        sampleTargetHit=not camera,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)
    if not camera:
        # only consider valid hits
        valid = result["hitSuccess"] == 1
        assert np.all(result["wavelengthHit"][valid] == lam)
        assert np.all(result["objectIdHit"][valid] == objectId)
    absorbed = result["hitResult"] == EventResultCode.RAY_ABSORBED
    assert np.all(result["hitResult"][~absorbed] == EventResultCode.RAY_HIT)

    normal = np.array(normal)
    cosNrm = np.multiply(result["directionOut"], normal[None, :]).sum(1)
    refl = (cosNrm >= 0.0) & ~absorbed
    trans = (cosNrm < 0.0) & ~absorbed
    assert trans.sum() == 0
    # calculate reflectance if required
    if R < 0.0:
        ni = waterModel.refractive_index(lam) * np.ones(N)
        no, k = np.ones(N) * n, np.ones(N) * k
        R_ = reflectance_metal(result["directionIn"], normal, ni, no, k)
    else:
        R_ = np.ones(N) * R
    r = reflect(result["directionIn"], normal)
    if not particle:
        c = result["contribIn"]

    # check we prevent self intersection
    cosPos = np.multiply(result["positionOut"], normal[None, :]).sum(-1)
    assert np.all(cosPos[refl] > 0.0)

    assert np.allclose(result["directionOut"][refl], r[refl], atol=1e-5)
    assert np.all(result["mediumIdxOut"][refl] == waterIdx)

    if not particle and not camera:
        hit = result["hitSuccess"] == 1
        cO = result["contribHit"]
        assert np.allclose(cO[hit], (c * (1.0 - R_))[hit], atol=5e-7)
        assert np.all(cO[hit] > 0.0)
        assert np.all((R_ >= 1.0)[~hit])
    if "R" in flags:
        if R != 0.0:
            assert refl.sum() > 0
        else:
            assert refl.sum() == 0
        if (particle or absorb) and R < 1.0:
            assert absorbed.sum() > 0
        elif R == 0.0:
            assert absorbed.sum() == N
        else:
            assert absorbed.sum() == 0
        if particle:
            # check particle is not both reflected and detected
            np.all((result["hitSuccess"] == 0) == refl)
        else:
            cO = result["contribOut"][~absorbed]
            cO_exp = c if absorb else c * R_
            assert np.allclose(cO, cO_exp[~absorbed])
    else:
        # no reflection -> absorb all
        assert np.all(result["hitResult"] == EventResultCode.RAY_ABSORBED)


@pytest.mark.parametrize(
    "particle, camera", [(True, False), (False, True), (False, False)]
)
def test_AbsorbingSurface(particle: bool, camera: bool):
    N = 32 * 1024
    lam = 600.0 * u.nm
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    surface = theia.surface.AbsorbingSurface()
    mat = Material("mat", None, water, surface)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = ConeCamera(
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            objectId=objectId,
        )
    else:
        source = ConeLightSource(
            photons,
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
            emitParticles=particle,
        )
        photons = None
    sampler = SurfaceInteractionSampler(
        N,
        ray,
        surface,
        matStore,
        rng,
        source,
        photons,
        material="mat",
        surfaceNormal=normal,
        objectId=objectId,
        sampleTargetHit=not camera,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)
    assert np.all(result["hitResult"] == EventResultCode.RAY_ABSORBED)

    if not camera:
        assert np.all(result["hitSuccess"] == 1)
        assert np.all(result["positionHit"] == result["positionIn"])
        if not particle:
            assert np.all(result["contribHit"] == result["contribIn"])


@pytest.mark.parametrize(
    "particle, camera", [(True, False), (False, True), (False, False)]
)
def test_BorderSurface(particle: bool, camera: bool):
    N = 32 * 1024
    lam = 600.0 * u.nm
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    surface = theia.surface.BorderSurface()
    mat = Material("mat", None, water, surface)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = ConeCamera(
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            objectId=objectId,
        )
    else:
        source = ConeLightSource(
            photons,
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
            emitParticles=particle,
        )
        photons = None
    sampler = SurfaceInteractionSampler(
        N,
        ray,
        surface,
        matStore,
        rng,
        source,
        photons,
        material="mat",
        surfaceNormal=normal,
        objectId=objectId,
        sampleTargetHit=not camera,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)
    assert np.all(result["hitResult"] == EventResultCode.VOLUME_HIT)
    if not camera:
        assert np.all(result["hitSuccess"] == 0)

    # check we actually crossed the border (prevent self intersection)
    assert np.all(result["mediumIdxOut"] == VACUUM_IDX)
    normal = np.array(normal)
    cosPos = np.multiply(result["positionOut"], normal[None, :]).sum(-1)
    assert np.all(cosPos < 0.0)


@pytest.mark.parametrize(
    "particle,camera,flags",
    [
        (True, False, "TR"),
        (True, False, "T"),
        (True, False, "R"),
        (True, False, "DR"),
        (False, True, "TR"),
        (False, True, "T"),
        (False, True, "R"),
        (False, False, "TR"),
        (False, False, "T"),
        (False, False, "R"),
        (False, False, "DR"),
    ],
)
def test_LambertianReflectingSurface(particle: bool, camera: bool, flags: str):
    N = 32 * 1024
    lam = 600.0 * u.nm
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10
    reflectivity = 0.8

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    surface = theia.surface.LambertianReflectingSurface()
    probs = {"reflectivity": TableProperty.createConstTable(reflectivity)}
    mat = Material("mat", None, water, surface, flags=flags, properties=probs)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = ConeCamera(
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            objectId=objectId,
        )
    else:
        source = ConeLightSource(
            photons,
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
            emitParticles=particle,
        )
        photons = None
    sampler = SurfaceInteractionSampler(
        N,
        ray,
        surface,
        matStore,
        rng,
        source,
        photons,
        material="mat",
        surfaceNormal=normal,
        objectId=objectId,
        sampleTargetHit=not camera,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)
    if not camera:
        # only consider valid hits
        valid = result["hitSuccess"] == 1
        assert np.all(result["wavelengthHit"][valid] == lam)
        assert np.all(result["objectIdHit"][valid] == objectId)
    canReflect = flags != "D" and flags != "T"
    if not canReflect:
        # no reflection -> absorb everything
        assert np.all(result["hitResult"] == EventResultCode.RAY_ABSORBED)
        return
    absorbed = result["hitResult"] == EventResultCode.RAY_ABSORBED
    assert np.all(result["hitResult"][~absorbed] == EventResultCode.RAY_HIT)

    # check contribution
    if particle:
        assert np.all(result["hitSuccess"][absorbed] == 1)
        assert np.all(result["hitSuccess"][~absorbed] == 0)
        assert abs(1.0 - absorbed.sum() / N - reflectivity) < 0.01  # noisy MC estimate
    else:
        if not camera:
            assert np.all(result["hitSuccess"] == 1)
            assert np.allclose(
                result["contribHit"], result["contribIn"] * (1.0 - reflectivity)
            )
        assert absorbed.sum() == 0
        assert np.allclose(
            result["contribIn"][~absorbed] * reflectivity,
            result["contribOut"][~absorbed],
        )

    # check we prevent self intersection
    normal = np.array(normal)
    cosPos = np.multiply(result["positionOut"], normal[None, :]).sum(-1)
    assert np.all(cosPos[~absorbed] > 0.0)  # all reflected
    assert np.all(result["mediumIdxOut"] == waterIdx)

    # check angular distribution of reflection
    # TODO: Be a bit more sophisticated here
    cosNrm = np.multiply(result["directionOut"][~absorbed], normal[None, :]).sum(1)
    assert cosNrm.min() > 0.0 and cosNrm.min() < 0.05
    assert cosNrm.max() > 0.95 and cosNrm.max() <= 1.0


@pytest.mark.parametrize(
    "camera,flags",
    [
        (True, "T"),
        (True, "R"),
        (True, "D"),
        (True, "TR"),
        (False, "T"),
        (False, "R"),
        (False, "D"),
        (False, "TR"),
    ],
)
def test_LambertianReflectingSurface_MIS(camera: bool, flags: str):
    N = 32 * 1024
    lam = 600.0 * u.nm
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10
    reflectivity = 0.8

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    surface = theia.surface.LambertianReflectingSurface()
    probs = {"reflectivity": TableProperty.createConstTable(reflectivity)}
    mat = Material("mat", None, water, surface, flags=flags, properties=probs)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    # create pipeline
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = ConeCamera(
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            objectId=objectId,
        )
    else:
        source = ConeLightSource(
            photons,
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
        )
        photons = None
    sampler = SurfaceScatteringSampler(
        N,
        ray,
        surface,
        matStore,
        rng,
        source,
        photons,
        material="mat",
        surfaceNormal=normal,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)

    assert np.allclose(result["probSampled"], result["probEval"])
    normal = np.array(normal)
    cosNrm = np.multiply(result["randomDir"], normal[None, :]).sum(-1)
    cosNrmSampled = np.multiply(result["sampledDir"], normal[None, :]).sum(-1)
    # TODO: more elaborate testing on the sampled distribution (also include phi...)
    assert cosNrmSampled.min() > 0.0 and cosNrmSampled.min() < 0.05
    assert cosNrmSampled.max() > 0.95 and cosNrmSampled.max() <= 1.0
    absorbed = result["scatterResult"] == EventResultCode.RAY_ABSORBED
    m = ~absorbed  # mask
    if flags == "T" or flags == "D":
        # cannot reflect -> no success
        assert np.all(absorbed)
        return  # nothing left to test
    assert np.all(~absorbed == (cosNrm > 0.0))
    assert np.all(absorbed == (cosNrm < 0.0))
    assert np.allclose(result["probRandom"][m], cosNrm[m] / np.pi)
    assert np.allclose(result["probRandom"][~m], 0.0)
    assert np.all(result["mediumIdxOut"][m] == waterIdx)
    expContrib = result["contribIn"][m] * reflectivity / np.pi
    assert np.allclose(result["contribOut"][m], expContrib)
    assert np.allclose(result["directionOut"][m], result["randomDir"][m])
    # check we prevent self intersection
    cosPos = np.multiply(result["positionOut"], normal[None, :]).sum(-1)
    assert np.all(cosPos[m] > 0.0)  # all reflected
