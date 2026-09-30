import pytest

import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl

from ctypes import c_float
from hephaistos.queue import QueueTensor, QueueBuffer

from theia.camera import ConeCamera, PencilCamera
from theia.compiler import compileShader, createPreamble
from theia.light import ConeLightSource, ConstWavelengthSource, PencilLightSource
from theia.material import Material, MaterialStore, VACUUM_IDX
from theia.material import Medium, MediumReferenceProperty
from theia.model import BK7Model, PureWaterModel
from theia.property import FloatProperty, Property, TableProperty
from theia.random import PhiloxRNG
from theia.ray import UnpolarizedRay
from theia.testing import SurfaceInteractionSampler, SurfaceScatteringSampler
from theia.trace import EventResultCode
from theia.util import createCType
import theia.units as u
import theia.surface


def reflect(i, n):
    ct = np.multiply(i, n[None, :]).sum(1)
    return i - 2.0 * ct[:, None] * n


def refract(i, n, ni, no):
    eta = ni / no
    ct = np.multiply(i, n[None, :]).sum(1)
    k = 1.0 - eta * eta * (1.0 - ct * ct)
    # k < 0 marks total internal reflection; clip to avoid a sqrt(nan) warning
    return eta[:, None] * i - (eta * ct + np.sqrt(np.clip(k, 0.0, None)))[:, None] * n


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


def fresnel_thin(ci, ni, nl, kl, no, d, lam):
    n1, n2, n3 = ni, nl + 1.0j * kl, no
    # snell's law
    c1 = np.abs(ci)
    s1 = np.sqrt(np.clip(1.0 - np.square(c1), 0.0, 1.0))
    c2 = np.sqrt(1.0 - (n1 / n2 * s1) ** 2, dtype=complex)
    c3 = np.sqrt(1.0 - (n1 / n3 * s1) ** 2, dtype=complex)
    # fresnel coefficients
    r12_s = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    r23_s = (n2 * c2 - n3 * c3) / (n2 * c2 + n3 * c3)
    r12_p = (n1 * c2 - n2 * c1) / (n1 * c2 + n2 * c1)
    r23_p = (n2 * c3 - n3 * c2) / (n2 * c3 + n3 * c2)
    t12_s = 2.0 * n1 * c1 / (n1 * c1 + n2 * c2)
    t23_s = 2.0 * n2 * c2 / (n2 * c2 + n3 * c3)
    t12_p = 2.0 * n1 * c1 / (n1 * c2 + n2 * c1)
    t23_p = 2.0 * n2 * c2 / (n3 * c2 + n2 * c3)
    # interference?
    if d is None:
        assert np.all(np.asarray(kl) == 0.0)  # model assumption
        norm = lambda c: c.real**2 + c.imag**2
        R12s, R12p = norm(r12_s), norm(r12_p)
        R23s, R23p = norm(r23_s), norm(r23_p)
        T12s, T12p = 1.0 - R12s, 1.0 - R12p
        # R12, R23 close to 1 will produce inf or NaN
        # Since R12 close to 1 means hardly any transmission (T12 ~ 0.0), replace by 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            Rs = R12s + np.nan_to_num(T12s**2 * R23s / (1.0 - R12s * R23s), posinf=0.0)
            Rp = R12p + np.nan_to_num(T12p**2 * R23p / (1.0 - R12p * R23p), posinf=0.0)
        R = 0.5 * (Rs + Rp)
        T = 1.0 - R
        A = np.zeros_like(R)
    else:
        beta = 2.0 * np.pi * d / lam * n2 * c2
        denom_s = 1.0 + r12_s * r23_s * np.exp(2.0j * beta)
        r123_s = (r12_s + r23_s * np.exp(2.0j * beta)) / denom_s
        t123_s = (t12_s * t23_s * np.exp(1.0j * beta)) / denom_s
        denom_p = 1.0 + r12_p * r23_p * np.exp(2.0j * beta)
        r123_p = (r12_p + r23_p * np.exp(2.0j * beta)) / denom_p
        t123_p = (t12_p * t23_p * np.exp(1.0j * beta)) / denom_p
        norm = lambda c: c.real**2 + c.imag**2
        Rs, Rp = norm(r123_s), norm(r123_p)
        t_scale = ((n3 * c3) / (n1 * c1)).real
        Ts, Tp = t_scale * norm(t123_s), t_scale * norm(t123_p)
        R, T = 0.5 * (Rs + Rp), 0.5 * (Ts + Tp)
        A = 1.0 - R - T
    # due to finite numerical precision, we might be slightly off
    # -> push into valid range (but check before to not hide any errors)
    assert np.all((R >= -5e-5) & (R <= (1.0 + 5e-5)))
    assert np.all((T >= -5e-5) & (T <= (1.0 + 5e-5)))
    assert np.all((A >= -5e-5) & (A <= (1.0 + 5e-5)))
    R = np.clip(R, 0.0, 1.0)
    T = np.clip(T, 0.0, 1.0)
    A = np.clip(A, 0.0, 1.0)
    return R, T, A


@pytest.mark.parametrize("interference", [True, False])
def test_fresnel_thinLayer(interference: bool):
    N = 32 * 64

    # allocate memory
    queue_fields = [
        ("n1", c_float),
        ("n2", c_float),
        ("k2", c_float),
        ("n3", c_float),
        ("d", c_float),
        ("cos0", c_float),
        ("lambda", c_float),
        ("Rs", c_float),
        ("Rp", c_float),
        ("Ts", c_float),
        ("Tp", c_float),
    ]
    queueItem = createCType("Item", queue_fields)
    queue_tensor = QueueTensor(queueItem, N, skipCounter=True)
    queue_buffer = QueueBuffer(queueItem, N, skipCounter=True)

    # create test program
    preamble = createPreamble(INCLUDE_INTERFERENCE=interference)
    program = hp.Program(compileShader("fresnel_thinLayer.test.glsl", preamble))
    program.bindParams(Result=queue_tensor)
    # run it
    (
        hp.beginSequence()
        .And(program.dispatch(32))
        .Then(hp.retrieveTensor(queue_tensor, queue_buffer))
        .Submit()
        .wait()
    )

    # check result
    result = queue_buffer.view
    d = result["d"] if interference else None
    R, T, A = fresnel_thin(
        result["cos0"],
        result["n1"],
        result["n2"],
        result["k2"],
        result["n3"],
        d,
        result["lambda"],
    )
    assert np.all((R >= 0.0) & (R <= 1.0))
    assert np.all((T >= 0.0) & (T <= 1.0))
    assert np.all((A >= 0.0) & (A <= 1.0))
    assert np.allclose(R + T + A, 1.0)
    dielectric = result["k2"] == 0.0
    assert np.allclose(A[dielectric], 0.0, atol=1e-7)
    R_ = 0.5 * (result["Rs"] + result["Rp"])
    T_ = 0.5 * (result["Ts"] + result["Tp"])
    assert np.allclose(R, R_, atol=5e-4)
    assert np.allclose(T, T_, atol=6e-4)


@pytest.mark.parametrize("interference", [True, False])
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
def test_ThinDielectricSurface(
    particle: bool, camera: bool, flags: str, interference: bool
):
    N = 32 * 1024
    lam = 600.0 * u.nm
    d = 20.0 * u.nm
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    glassModel = BK7Model()
    glass = glassModel.createMedium()
    props: dict[str, Property] = {
        "layer_thickness": FloatProperty(d),
        "layer_medium": MediumReferenceProperty(glass),
    }
    surface = theia.surface.ThinDielectricSurface(interference=interference)
    mat = Material("mat", None, water, surface, flags=flags, properties=props)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]
    # fetch refractive indices for later
    ni = waterModel.refractive_index(lam)
    nl = glassModel.refractive_index(lam)
    no = 1.0  # vacuum

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
    if not interference:
        d = None
    cosIn = np.multiply(result["directionIn"], normal[None, :]).sum(-1)
    R, T, A = fresnel_thin(cosIn, ni, nl, 0.0, no, d, lam)
    ni, no = np.ones(N) * ni, np.ones(N) * no  # cast to array for following funcs
    t = refract(result["directionIn"], normal, ni, no)
    r = reflect(result["directionIn"], normal)
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
            assert np.allclose(cO[~absorbed], (c * R)[~absorbed], atol=5e-4, rtol=5e-4)
        if not particle and not camera:
            cH = result["contribHit"]
            assert np.allclose(cH[~absorbed], (c * T)[~absorbed], atol=5e-4)
    if flags == "T":
        assert refl.sum() == 0
        assert absorbed.sum() > 0  # total internal reflection
        if not particle:
            cO = result["contribOut"]
            # TODO: Check why this large error!?
            assert np.allclose(cO[~absorbed], (c * T)[~absorbed], atol=1e-3, rtol=5e-4)
        if not particle and not camera:
            cH = result["contribHit"]
            assert np.allclose(cH[~absorbed], (c * T)[~absorbed], atol=5e-4)
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
            assert np.allclose(cH[~absorbed], (c * T)[~absorbed], atol=5e-4)


@pytest.mark.parametrize("absorb", [True, False])
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
def test_ThinMetallicSurface(particle: bool, camera: bool, flags: str, absorb: bool):
    N = 32 * 1024
    lam = 600.0 * u.nm
    n, k = 1.9, 2.35
    d = 20.0 * u.nm
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    metal = createConstMetal("metal", n, k)
    props: dict[str, Property] = {
        "layer_thickness": FloatProperty(d),
        "layer_medium": MediumReferenceProperty(metal),
    }
    surface = theia.surface.ThinMetallicSurface(absorb=absorb)
    mat = Material("mat", None, water, surface, flags=flags, properties=props)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]
    # fetch refractive indices for later
    ni = waterModel.refractive_index(lam)
    no = 1.0  # vacuum

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
    cosIn = np.multiply(result["directionIn"], normal[None, :]).sum(-1)
    R, T, A = fresnel_thin(cosIn, ni, n, k, no, d, lam)
    trans = (cosNrm < 0.0) & ~absorbed
    refl = (cosNrm >= 0.0) & ~absorbed
    ni, no = np.ones(N) * ni, np.ones(N) * no  # cast to array for following funcs
    t = refract(result["directionIn"], normal, ni, 1.0)
    r = reflect(result["directionIn"], normal)
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
        if not particle and not camera:
            cH = result["contribHit"]
            assert np.allclose(cH[~absorbed], (c * A)[~absorbed], atol=5e-4)
    if flags == "T":
        assert refl.sum() == 0
        assert absorbed.sum() > 0  # total internal reflection
        if not particle:
            cO = result["contribOut"]
            assert np.allclose(cO[~absorbed], (c * T)[~absorbed], atol=5e-4)
        if not particle and not camera:
            cH = result["contribHit"]
            assert np.allclose(cH[~absorbed], (c * A)[~absorbed], atol=5e-4)
    if flags == "TR":
        assert refl.sum() > 0
        assert trans.sum() > 0
        if absorb or particle:
            assert absorbed.sum() > 0
        else:
            assert absorbed.sum() == 0
        if not particle and not camera:
            cH = result["contribHit"]
            assert np.allclose(cH[~absorbed], (c * A)[~absorbed], rtol=2e-4)
        if particle:
            # check particle is not both reflected and detected
            np.all((result["hitSuccess"] == 0) == refl)
        else:
            if not absorb:
                # apply IS correction from sampling without absorption
                c *= R + T
            assert np.allclose(result["contribOut"], c, atol=5e-4, rtol=5e-4)


def reflect_arr(i, n):
    ct = np.sum(i * n, axis=1)
    return i - 2.0 * ct[:, None] * n

def refract_arr(i, n, ni, no):
    eta = ni / no
    ct = np.sum(i * n, axis=1)
    k = 1.0 - eta * eta * (1.0 - ct * ct)
    # k < 0 marks total internal reflection; clip to avoid a sqrt(nan) warning
    return eta[:, None] * i - (eta * ct + np.sqrt(np.clip(k, 0.0, None)))[:, None] * n

def reflectance_arr(i, n, ni, no):
    ci = np.abs(np.sum(i * n, axis=1))
    si = np.sqrt(np.clip(1.0 - np.square(ci), 0.0, 1.0))
    so = si * ni / no
    co = np.sqrt(np.clip(1.0 - np.square(so), 0.0, 1.0))
    rs = (ni * ci - no * co) / (ni * ci + no * co)
    rp = (no * ci - ni * co) / (no * ci + ni * co)
    return 0.5 * (rs * rs + rp * rp)

def _sample_theta_from_pdf(theta_grid, pdf, N):
    """Draw N polar angles by numerically integrating `pdf` on `theta_grid` into a
    CDF and inverting it (inverse-transform sampling via a tabulated CDF)."""
    cdf = np.cumsum(pdf)
    cdf -= cdf[0]
    cdf /= cdf[-1]
    u = np.random.uniform(0.0, 1.0, N)
    return np.interp(u, cdf, theta_grid)


def _build_microfacet_normals(cos_theta, normals, tangential_vector, N):
    """Rotate the surface normals by (theta, phi) with phi uniform, giving isotropic
    micro-facet normals for a given polar-angle distribution `cos_theta`."""
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    phi = np.random.uniform(0.0, 2.0 * np.pi, N)
    third_vectors = np.cross(normals, tangential_vector[None, :])
    return (
        cos_theta[:, None] * normals
        + (sin_theta * np.cos(phi))[:, None] * tangential_vector[None, :]
        + (sin_theta * np.sin(phi))[:, None] * third_vectors
    )


def sample_microfacet_normals_beckmann(normals, tangential_vector, direction, alpha, N, numSamples=8192):
    """
    Sample N microfacet normals from the Beckmann NDF.

    Rather than copying the shader's closed-form inverse, the polar-angle pdf is built 
    directly from the NDF as an independent ground truth,
        p(theta) ~ D(theta) * cos(theta) * sin(theta),
        D(theta) = exp(-tan^2(theta) / alpha^2) / (pi * alpha^2 * cos^4(theta)),
    numerically integrated into a CDF and inverted. 
    """
    theta_grid = np.linspace(0.0, np.pi / 2.0, numSamples)
    cos_t = np.cos(theta_grid)
    sin_t = np.sin(theta_grid)
    # guard the cos->0 singularity at grazing theta (pdf vanishes there anyway)
    valid = cos_t > 1e-6
    cos_safe = np.where(valid, cos_t, 1.0)
    tan2 = (sin_t / cos_safe) ** 2
    D = np.exp(-tan2 / alpha**2) / (np.pi * alpha**2 * cos_safe**4)
    pdf = np.where(valid, D * cos_t * sin_t, 0.0)
    theta = _sample_theta_from_pdf(theta_grid, pdf, N)
    return _build_microfacet_normals(np.cos(theta), normals, tangential_vector, N)


def sample_microfacet_normals_trowbridge_reitz(normals, tangential_vector, direction, alpha, N, numSamples=8192):
    """
    Sample N microfacet normals from the Trowbridge-Reitz (GGX) NDF.

    Rather than copying the shader's closed-form inverse, the polar-angle pdf is built 
    directly from the NDF as an independent ground truth,
        p(theta) ~ D(theta) * cos(theta) * sin(theta),
        D(theta) = alpha^2 / (pi * cos^4(theta) * (alpha^2 + tan^2(theta))^2),
    numerically integrated into a CDF and inverted.
    """
    theta_grid = np.linspace(0.0, np.pi / 2.0, numSamples)
    cos_t = np.cos(theta_grid)
    sin_t = np.sin(theta_grid)
    valid = cos_t > 1e-6
    cos_safe = np.where(valid, cos_t, 1.0)
    tan2 = (sin_t / cos_safe) ** 2
    D = alpha**2 / (np.pi * cos_safe**4 * (alpha**2 + tan2) ** 2)
    pdf = np.where(valid, D * cos_t * sin_t, 0.0)
    theta = _sample_theta_from_pdf(theta_grid, pdf, N)
    return _build_microfacet_normals(np.cos(theta), normals, tangential_vector, N)


def sample_microfacet_normals_gaussian(normals, tangential_vector, direction, alpha, N, numSamples=4096):
    """
    Sample N microfacet normals from the Gaussian slope distribution used by the
    "gaussian" model: 
        p(theta) ~ sin(theta) * exp(-theta^2 / (2 alpha^2))
    """
    theta_grid = np.linspace(0.0, np.pi / 2.0, numSamples)
    pdf = np.sin(theta_grid) * np.exp(-theta_grid**2 / (2.0 * alpha**2))
    theta = _sample_theta_from_pdf(theta_grid, pdf, N)
    return _build_microfacet_normals(np.cos(theta), normals, tangential_vector, N)


def _perpendicular_to_vector(v):
    """Unit vector perpendicular to v."""
    s = 1.0 if v[2] >= 0.0 else -1.0
    a = -1.0 / (s + v[2])
    b = v[0] * v[1] * a
    return np.array([b, s + v[1] * v[1] * a, -v[1]])


def _perpendicular_to_pair(a, b):
    """Normalize(cross(a, b)); falls back to _perpendicular_to_vector(a) if degenerate."""
    c = np.cross(a, b)
    length = np.linalg.norm(c)
    return c / length if length >= 1e-5 else _perpendicular_to_vector(a)


def _sample_unit_disk(N):
    """Concentric disk sampling. Returns (N, 2) array."""
    u = np.random.uniform(0.0, 1.0, (N, 2))
    rng = 2.0 * u - 1.0
    degenerate = (rng[:, 0] == 0.0) & (rng[:, 1] == 0.0)
    cond = np.abs(rng[:, 0]) > np.abs(rng[:, 1])
    safe_x = np.where(np.abs(rng[:, 0]) < 1e-30, 1.0, rng[:, 0])
    safe_y = np.where(np.abs(rng[:, 1]) < 1e-30, 1.0, rng[:, 1])
    r = np.where(cond, rng[:, 0], rng[:, 1])
    phi = np.where(
        cond,
        (np.pi / 4.0) * rng[:, 1] / safe_x,
        (np.pi / 2.0) - (np.pi / 4.0) * rng[:, 0] / safe_y,
    )
    px = np.where(degenerate, 0.0, r * np.cos(phi))
    py = np.where(degenerate, 0.0, r * np.sin(phi))
    return np.column_stack([px, py])


def _masking_function_trowbridge_reitz(cos_n, alpha):
    """Smith masking for Trowbridge-Reitz: 1 / (1 + Lambda)."""
    valid = cos_n >= 1e-3
    cos_n_safe = np.where(valid, cos_n, 1.0)
    tan2_n = np.maximum(1.0 - cos_n_safe**2, 0.0) / (cos_n_safe**2)
    lam = (np.sqrt(1.0 + alpha**2 * tan2_n) - 1.0) / 2.0
    return np.where(valid, 1.0 / (1.0 + lam), 0.0)


def sample_microfacet_normals_trowbridge_reitz_shadowed(normals, tangential_vector, direction, alpha, N):
    """
    Sample N microfacet normals from the VNDF for Trowbridge-Reitz.

    """
    # pencil beam: all normals identical
    normal = normals[0]  

    # Transformation matrix for local coordinate system. The surface normal is the new z-axis, and the
    # tangential component of the incoming ray is along the new y-axis.
    vx = _perpendicular_to_pair(normal, direction)
    vy = np.cross(normal, vx)
    trafo = np.column_stack([vx, vy, normal])

    cos_n = float(abs(np.dot(normal, direction)))
    sin_n = np.sqrt(max(1.0 - cos_n**2, 0.0))

    # transform surface normal to hemispherical configuration
    wh = np.array([0.0, alpha * sin_n, cos_n])
    wh_len = np.linalg.norm(wh)
    if wh_len > 1e-12:
        wh /= wh_len

    # Orthonormal basis for disk sampling
    T1 = np.array([-1.0, 0.0, 0.0])
    T2 = np.cross(wh, T1) 

    # sample point on unit disk
    p = _sample_unit_disk(N)  # (N, 2)
    px, py = p[:, 0], p[:, 1]

    # warp hemispherical projection for visible normal sampling
    h = np.sqrt(np.maximum(1.0 - px**2, 0.0))
    t = (1.0 + wh[2]) / 2.0
    py_warped = h * (1.0 - t) + py * t

    # Reproject to hemisphere and transform normal to ellipsoid configuration
    pz = np.sqrt(np.maximum(0.0, 1.0 - px**2 - py_warped**2))

    nh = (
        px[:, None] * T1[None, :]
        + py_warped[:, None] * T2[None, :]
        + pz[:, None] * wh[None, :]
    )

    mn_local = np.column_stack([
        alpha * nh[:, 0],
        alpha * nh[:, 1],
        np.maximum(1e-6, nh[:, 2]),
    ])
    mn_local /= np.linalg.norm(mn_local, axis=1, keepdims=True)

    # transform from local to global coordinate system
    return mn_local @ trafo.T # (N, 3)


_ROUGH_DIELECTRIC_PARAMS = [
    (True, False, "TR", 0),
    (True, False, "TR", 20),
    (True, False, "TR", 80),
    (True, False, "T", 40),
    (True, False, "R", 40),
    (True, False, "DR", 40),
    (False, True, "TR", 0),
    (False, True, "TR", 20),
    (False, True, "TR", 80),
    (False, True, "T", 40),
    (False, True, "R", 40),
    (False, False, "TR", 0),
    (False, False, "TR", 20),
    (False, False, "TR", 80),
    (False, False, "T", 40),
    (False, False, "R", 40),
    (False, False, "DR", 40),
]


def _test_rough_dielectric_surface(surface, microfacet_sampler, particle, camera, flags, angle, alpha=0.15, masking_function=None):
    np.random.seed(0xABCD + 10*(1+particle+2*camera)*int(angle))
    N = 64 * 1024
    lam = 600.0 * u.nm
    direction = (
        np.array((0.8, 0.36, 0.48)) * np.cos(np.deg2rad(angle))
        + np.array((-0.36, 0.8, 0)) * np.sin(np.deg2rad(angle)) / np.sqrt(481 / 625)
    )
    normal = np.array((-0.8, -0.36, -0.48)).astype(np.float32)
    tangential_vector = np.array((-0.36, 0.8, 0)) / np.sqrt(481 / 625)
    objectId = 10

    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    properties = {"roughness_parameter": FloatProperty(alpha)}
    mat = Material("mat", None, water, surface, flags=flags, properties=properties)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA + 10*(1+particle+camera)*int(angle))
    photons = ConstWavelengthSource(lam)
    if camera:
        source = PencilCamera(rayDirection=direction, mediumIdx=waterIdx, objectId=objectId)
    else:
        source = PencilLightSource(
            photons,
            direction=direction,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
            emitParticles=particle,
        )
        photons = None
    sampler = SurfaceInteractionSampler(
        N, ray, surface, matStore, rng, source, photons,
        material="mat", surfaceNormal=normal, objectId=objectId,
        sampleTargetHit=not camera,
    )
    pl.runPipeline(sampler.collectStages())

    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)
    if not camera:
        assert np.allclose(result["directionIn"], direction, atol=1e-6)
        valid = result["hitSuccess"] == 1
        assert np.all(result["wavelengthHit"][valid] == lam)
        assert np.all(result["objectIdHit"][valid] == objectId)
    if flags == "D":
        assert np.all(result["hitResult"] == EventResultCode.RAY_ABSORBED)
        return
    absorbed = result["hitResult"] == EventResultCode.RAY_ABSORBED
    assert np.all(result["hitResult"][~absorbed] == EventResultCode.RAY_HIT)

    normals = np.tile(normal, (N, 1)).astype(np.float32)
    cosNrm = np.multiply(result["directionOut"], normal[None, :]).sum(1).astype(np.float32)
    trans = (cosNrm < 0.0) & ~absorbed
    refl = (cosNrm > 0.0) & ~absorbed
    ni = waterModel.refractive_index(lam) * np.ones(N)
    no = np.ones(N)

    # check we prevent self intersection
    cosPos = np.multiply(result["positionOut"], normal[None, :]).sum(-1)
    assert np.all(cosPos[trans] < 0.0)
    assert np.all(cosPos[refl] > 0.0)
    assert np.all(result["mediumIdxOut"][trans] == VACUUM_IDX)
    assert np.all(result["mediumIdxOut"][refl] == waterIdx)

    if not particle:
        c = result["contribIn"]
    if camera:
        eta = ni / no
        c = np.copy(result["contribIn"])
        c[trans] = (c * eta * eta)[trans]

    if microfacet_sampler is None:
        # no sampler: geometric checks only
        if flags == "R" or flags == "DR":
            assert trans.sum() == 0
        if flags == "T":
            assert refl.sum() == 0
        if flags == "TR":
            assert absorbed.sum() == 0
        return

    ref_normals = microfacet_sampler(normals, tangential_vector, direction, alpha, N).astype(np.float32)
    t = refract_arr(result["directionIn"], ref_normals, ni, no).astype(np.float32)
    r = reflect_arr(result["directionIn"], ref_normals).astype(np.float32)
    R = reflectance_arr(result["directionIn"], ref_normals, ni, no).astype(np.float32)
    cos_in = np.sum(result["directionIn"] * ref_normals, axis=1).astype(np.float32)
    cos_t = np.sum(t * normal[None, :], axis=1).astype(np.float32)
    cos_r = np.sum(r * normal[None, :], axis=1).astype(np.float32)
    if masking_function is not None:
        # VNDF sampling guarantees the incoming ray hits the microfacet from the front;
        # apply stochastic masking to the outgoing direction only.
        u_r = np.random.uniform(0.0, 1.0, N)
        u_t = np.random.uniform(0.0, 1.0, N)
        mask_r = (cos_r > 0) & (masking_function(np.abs(cos_r), alpha) > u_r)
        mask_t = (cos_t < 0) & (R < 1) & (masking_function(np.abs(cos_t), alpha) > u_t)
    else:
        mask_r = (cos_in < 0) & (cos_r > 0)
        mask_t = (cos_in < 0) & (cos_t < 0) & (R < 1)

    # use relative error for reflectivities / angles and absolute errors for components of
    # directions (they can be very close to 0 for some incident angles)
    # Note: These tolerances are not super high considering the amount of tests we run. There
    # is a non-zero chance that some tests will fail if the RNG seed gets changed.
    rel_err = 0.015
    abs_err = 0.004

    # Direction comparisons are only reliable when there are enough reference samples.
    enough_t = (mask_t.sum() >= 20000) and (trans.sum() >= 20000)
    enough_r = (mask_r.sum() >= 20000) and (refl.sum() >= 20000)

    # check correct sampling of micro-facets. The reflect-and-transmit behaviour
    # weights the accepted facet by the Fresnel reflectance, which a restricted
    # surface carries on the contribution, so both sides have to be weighted.
    if (flags == "R" or flags == "DR") and not particle and enough_r:
        microfacet_normals_shader = result["directionOut"] - result["directionIn"]
        microfacet_normals_shader /= np.linalg.norm(microfacet_normals_shader, axis=1, keepdims=True)
        cos_test = np.clip(np.sum(normals[mask_r] * ref_normals[mask_r], axis=1), 0.0, 1.0)
        cos_shader = np.clip(np.sum(normals * microfacet_normals_shader, axis=1), 0.0, 1.0)
        assert np.average(
            np.arccos(cos_test), weights=R[mask_r]
        ) == pytest.approx(
            np.average(np.arccos(cos_shader[refl]), weights=result["contribOut"][refl]),
            rel=rel_err,
        )

    # Every flag combination reproduces the reflect-and-transmit behaviour, so the
    # outgoing distribution is always Fresnel weighted. Restricted surfaces carry
    # that weight on the contribution instead of in the acceptance, so their
    # directions have to be compared contribution weighted.
    qR = R * mask_r
    qT = (1.0 - R) * mask_t
    # the queue has no contribution field for particle rays
    cO = result["contribOut"] if not particle else None
    if enough_t and (flags == "TR" or particle):
        assert np.mean(result["directionOut"][trans], axis=0) == pytest.approx(
            np.average(t[mask_t], weights=(1 - R[mask_t]), axis=0), abs=abs_err
        )
    if enough_t and flags == "T" and not particle:
        assert np.average(
            result["directionOut"][trans], weights=cO[trans], axis=0
        ) == pytest.approx(
            np.average(t[mask_t], weights=(1 - R[mask_t]), axis=0), abs=abs_err
        )
    if enough_r and (flags == "TR" or particle):
        assert np.mean(result["directionOut"][refl], axis=0) == pytest.approx(
            np.average(r[mask_r], weights=R[mask_r], axis=0), abs=abs_err
        )
    if enough_r and (flags == "R" or flags == "DR") and not particle:
        assert np.average(
            result["directionOut"][refl], weights=cO[refl], axis=0
        ) == pytest.approx(
            np.average(r[mask_r], weights=R[mask_r], axis=0), abs=abs_err
        )

    if flags == "R" or flags == "DR":
        assert trans.sum() == 0
        if particle and enough_r:
            assert absorbed.sum() > 0
        elif enough_r:
            # The reflected fraction of the reflect-and-transmit surface. The
            # weight is an expectation over *all* rays, and an absorbed ray
            # contributes zero, so compare the sum rather than the mean over the
            # survivors.
            expected = qR.mean() / (qR.mean() + qT.mean())
            assert cO[~absorbed].sum() / N == pytest.approx(
                expected * np.mean(c[~absorbed]), rel=rel_err
            )
    if flags == "T":
        assert refl.sum() == 0
        if particle and enough_t:
            absorbed.sum() > 0
        if not particle and enough_t:
            expected = qT.mean() / (qR.mean() + qT.mean())
            assert cO[~absorbed].sum() / N == pytest.approx(
                expected * np.mean(c[~absorbed]), rel=rel_err
            )
        if not particle and not camera and enough_t:
            # nothing is reflected back, so the detector sees everything
            cH = result["contribHit"]
            assert np.mean(cH[~absorbed]) == pytest.approx(
                np.mean(c[~absorbed]), rel=rel_err
            )
    if flags == "TR":
        if enough_r:
            assert refl.sum() > 0
        if enough_t:
            assert trans.sum() > 0
        assert absorbed.sum() == 0
        if particle:
            np.all((result["hitSuccess"] == 0) == refl)
        else:
            assert np.allclose(result["contribOut"], c)


@pytest.mark.parametrize("particle,camera,flags,angle", _ROUGH_DIELECTRIC_PARAMS)
def test_DielectricBeckmannSurface(particle: bool, camera: bool, flags: str, angle: float):
    _test_rough_dielectric_surface(
        theia.surface.DielectricRoughSurface(model="beckmann"),
        sample_microfacet_normals_beckmann,
        particle, camera, flags, angle,
    )


@pytest.mark.parametrize("particle,camera,flags,angle", _ROUGH_DIELECTRIC_PARAMS)
def test_DielectricTrowbridgeReitzSurface(particle: bool, camera: bool, flags: str, angle: float):
    _test_rough_dielectric_surface(
        theia.surface.DielectricRoughSurface(model="trowbridge_reitz"),
        sample_microfacet_normals_trowbridge_reitz,
        particle, camera, flags, angle,
    )


@pytest.mark.parametrize("particle,camera,flags,angle", _ROUGH_DIELECTRIC_PARAMS)
def test_DielectricTrowbridgeReitzShadowedSurface(particle: bool, camera: bool, flags: str, angle: float):
    _test_rough_dielectric_surface(
        theia.surface.DielectricRoughSurface(model="trowbridge_reitz_shadowed"),
        sample_microfacet_normals_trowbridge_reitz_shadowed,
        particle, camera, flags, angle,
        masking_function=_masking_function_trowbridge_reitz,
    )


@pytest.mark.parametrize("particle,camera,flags,angle", _ROUGH_DIELECTRIC_PARAMS)
def test_DielectricGaussianSurface(particle: bool, camera: bool, flags: str, angle: float):
    _test_rough_dielectric_surface(
        theia.surface.DielectricRoughSurface(model="gaussian"),
        sample_microfacet_normals_gaussian,
        particle, camera, flags, angle,
    )


# All four lobe weights must be present so their material slots are defined;
# the shader only enables the lobe split when all of them exist.
_LOBE_ZERO = {
    "prob_backscatter": 0.0,
    "prob_specularspike": 0.0,
    "prob_specularlobe": 0.0,
    "prob_diffuselobe": 0.0,
}


def _run_rough_reflection_lobe(lobe_probs, angle=30.0, alpha=0.15, N=32 * 1024):
    """Run the Geant4 UNIFIED surface in reflection-only mode ("R") with a forced
    reflection lobe and return the outgoing directions. Only the reflected
    directions are of interest here, so the surface cannot transmit. The ray
    travels in water towards vacuum.

    The lobe decomposition exists only in the UNIFIED model; the other rough
    models are a pure specular lobe."""
    lam = 600.0 * u.nm
    direction = (
        np.array((0.8, 0.36, 0.48)) * np.cos(np.deg2rad(angle))
        + np.array((-0.36, 0.8, 0)) * np.sin(np.deg2rad(angle)) / np.sqrt(481 / 625)
    )
    normal = np.array((-0.8, -0.36, -0.48)).astype(np.float32)
    objectId = 10

    properties = {"roughness_parameter": FloatProperty(alpha)}
    properties.update({k: FloatProperty(v) for k, v in lobe_probs.items()})
    water = PureWaterModel().createMedium()
    surface = theia.surface.DielectricRoughSurface(model="unified")
    mat = Material("mat", None, water, surface, flags="R", properties=properties)
    matStore = MaterialStore([mat])
    mediumIdx = matStore.media["water"]

    ray = UnpolarizedRay(particle=False)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    source = PencilLightSource(
        photons, direction=direction, mediumIdx=mediumIdx, timeRange=(0.0, 0.0)
    )
    sampler = SurfaceInteractionSampler(
        N, ray, surface, matStore, rng, source, None,
        material="mat", surfaceNormal=normal, objectId=objectId, sampleTargetHit=True,
    )
    pl.runPipeline(sampler.collectStages())

    result = sampler.queue.view(0)
    absorbed = result["hitResult"] == EventResultCode.RAY_ABSORBED
    return result, np.asarray(direction), normal, absorbed


# The UNIFIED walk decides reflection versus transmission by the Fresnel coin at
# the facet, so a reflection-only dielectric absorbs whatever the walk transmits -
# there is no analytic reflected fraction to weight the ray with instead. Only a
# few percent survive here (water to vacuum at 30 deg); how many exactly is the
# walk's business and not what these tests are about, they only need enough
# samples left to say something about the direction.
_LOBE_DIELECTRIC_MIN_REFLECTED = 1000


def test_DielectricRoughSurface_specularSpikeLobe():
    # specular spike -> deterministic reflection off the macroscopic surface normal
    result, direction, normal, absorbed = _run_rough_reflection_lobe(
        {**_LOBE_ZERO, "prob_specularspike": 1.0}
    )
    assert (~absorbed).sum() > _LOBE_DIELECTRIC_MIN_REFLECTED
    expected = reflect_arr(direction[None, :], normal[None, :])[0]
    assert np.allclose(result["directionOut"][~absorbed], expected[None, :], atol=1e-6)


def test_DielectricRoughSurface_backscatterLobe():
    # backscatter -> deterministic retro-reflection into the incoming direction
    result, direction, normal, absorbed = _run_rough_reflection_lobe(
        {**_LOBE_ZERO, "prob_backscatter": 1.0}
    )
    assert (~absorbed).sum() > _LOBE_DIELECTRIC_MIN_REFLECTED
    assert np.allclose(result["directionOut"][~absorbed], -direction[None, :], atol=1e-6)


def test_DielectricRoughSurface_diffuseLobe():
    # diffuse lobe -> cosine-weighted hemisphere about the surface normal,
    # tested exactly like the Lambertian reflecting surface
    result, direction, normal, absorbed = _run_rough_reflection_lobe(
        {**_LOBE_ZERO, "prob_diffuselobe": 1.0}
    )
    assert (~absorbed).sum() > _LOBE_DIELECTRIC_MIN_REFLECTED
    cosNrm = np.multiply(result["directionOut"][~absorbed], normal[None, :]).sum(1)
    assert np.all(cosNrm > 0.0)  # reflected back into the original hemisphere
    assert cosNrm.min() > 0.0 and cosNrm.min() < 0.05
    assert cosNrm.max() > 0.95 and cosNrm.max() <= 1.0
