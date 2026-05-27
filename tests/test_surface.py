import pytest

import hephaistos.pipeline as pl
import numpy as np

from theia.camera import ConeCamera, PencilCamera
from theia.light import ConeLightSource, ConstWavelengthSource, PencilLightSource
from theia.material import Material, MaterialStore, VACUUM_IDX
from theia.model import BK7Model, PureWaterModel
from theia.property import GaussianPPFTableProperty, TableProperty
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


def reflect_arr(i, n):
    ct = np.sum(i * n, axis=1)
    return i - 2.0 * ct[:, None] * n

def refract_arr(i, n, ni, no):
    eta = ni / no
    ct = np.sum(i * n, axis=1)
    k = 1.0 - eta * eta * (1.0 - ct * ct)
    return eta[:, None] * i - (eta * ct + np.sqrt(k))[:, None] * n

def reflectance_arr(i, n, ni, no):
    ci = np.abs(np.sum(i * n, axis=1))
    si = np.sqrt(np.clip(1.0 - np.square(ci), 0.0, 1.0))
    so = si * ni / no
    co = np.sqrt(np.clip(1.0 - np.square(so), 0.0, 1.0))
    rs = (ni * ci - no * co) / (ni * ci + no * co)
    rp = (no * ci - ni * co) / (no * ci + ni * co)
    return 0.5 * (rs * rs + rp * rp)

def sample_microfacet_normals(normals, tangential_vector, sigma):
    N = len(normals)
    alpha = np.random.normal(0, sigma, N)
    phi = np.random.uniform(0, np.pi, N)
    third_vectors = np.cross(normals, tangential_vector[None, :])
    ca, sa, cp, sp = np.cos(alpha), np.sin(alpha), np.cos(phi), np.sin(phi)
    return (normals * ca[:, None]
            + tangential_vector[None, :] * (sa * cp)[:, None]
            + third_vectors * (sa * sp)[:, None])


@pytest.mark.parametrize(
    "particle,camera,flags,angle",
    [
        (True, False, "TR",0),
        (True, False, "TR",30),
        (True, False, "TR",85),
        (True, False, "T",45),
        (True, False, "R",45),
        (True, False, "DR",45),
        (False, True, "TR",0),
        (False, True, "TR",30),
        (False, True, "TR",85),
        (False, True, "T",45),
        (False, True, "R",45),
        (False, False, "TR",0),
        (False, False, "TR",30),
        (False, False, "TR",85),
        (False, False, "T",45),
        (False, False, "R",45),
        (False, False, "DR",45),
    ],
)
def test_MicroFacetSurfaceModel(particle: bool, camera: bool, flags: str, angle: float):
    N = 32 * 1024
    lam = 600.0 * u.nm
    direction = np.array((0.8, 0.36, 0.48)) * np.cos(np.deg2rad(angle)) + np.array((-0.36, 0.8, 0)) * np.sin(np.deg2rad(angle)) / np.sqrt(481/625)
    normal = np.array((-0.8, -0.36, -0.48))
    tangential_vector = np.array((-0.36, 0.8, 0)) / np.sqrt(481/625)
    objectId = 10
    sigma = 0.0
    ppf = GaussianPPFTableProperty(0, sigma, 1024)

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    surface = theia.surface.MicroFacetSurfaceModel()
    mat = Material("mat", None, water, surface, flags=flags, properties={"ppf" : ppf})
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = PencilCamera(
            rayDirection=direction,
            mediumIdx=waterIdx,
            objectId=objectId,
        )
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
        assert np.allclose(result["directionIn"], direction, atol=1e-6)
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

    normals = np.tile(normal, (N,1))
    cosNrm = np.multiply(result["directionOut"], normal[None, :]).sum(1)
    trans = (cosNrm < 0.0) & ~absorbed
    refl = (cosNrm >= 0.0) & ~absorbed
    ni = waterModel.refractive_index(lam) * np.ones(N)
    no = np.ones(N)

    # sample micro-facets
    microfacet_normals = sample_microfacet_normals(normals, tangential_vector, sigma).astype(np.float32)
    t = refract_arr(result["directionIn"], microfacet_normals, ni, no).astype(np.float32)
    r = reflect_arr(result["directionIn"], microfacet_normals).astype(np.float32)
    R = reflectance_arr(result["directionIn"], microfacet_normals, ni, no).astype(np.float32)
    # check for valid micro-facets
    cos_in = np.multiply(result["directionIn"], microfacet_normals).sum(1).astype(np.float32)
    cos_t = np.multiply(t, normal[None, :]).sum(1).astype(np.float32)
    cos_r = np.multiply(r, normal[None, :]).sum(1).astype(np.float32)
    mask_r = (cos_in < 0) & (cos_r > 0)
    mask_t = (cos_in < 0) & (cos_t < 0) & (R < 1)
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

    err = 0.01 # allowed relative error for all probabilistic tests

    #check correct sampling of micro-facets
    if ((flags == "R" or flags == "DR") and not particle):
        microfacet_normals_shader = result["directionOut"] - result["directionIn"]
        microfacet_normals_shader /= np.linalg.norm(
            microfacet_normals_shader,
            axis=1,
            keepdims=True
        )
        cos_test = np.clip(np.sum(normals[mask_r] * microfacet_normals[mask_r], axis=1),0.0,1.0)
        cos_shader = np.clip(np.sum(normals * microfacet_normals_shader, axis=1),0.0,1.0)
        assert np.mean(np.arccos(cos_test)) == pytest.approx(np.mean(np.arccos(cos_shader)),rel=err)
        assert np.std(np.arccos(cos_test)) == pytest.approx(np.std(np.arccos(cos_shader)),rel=err)
       
    R_mean_r = np.mean(R[mask_r])
    R_mean_t = np.mean(R[mask_t])
    
    # Compare mean and std_dev of transmission/reflection. In cases where the program chooses between transmission/reflection
    # randomly, we have to weight the direction with the transmittance/reflectance (probability that ray gets transmitted/reflected)
    if (1 - R_mean_t) > 0.03 and (flags == "TR" or (flags == "T" and particle)):
        assert np.mean(result["directionOut"][trans],axis=0) == pytest.approx(np.average(t[mask_t],weights=(1-R[mask_t]),axis=0),rel=err)
    if (1 - R_mean_t) > 0.03 and (flags == "T") and not particle:
        assert np.mean(result["directionOut"][trans],axis=0) == pytest.approx(np.average(t[mask_t],axis=0),rel=err)
        assert np.std(result["directionOut"][trans],axis=0) == pytest.approx(np.std(t[mask_t],axis=0),rel=err)
    if R_mean_r > 0.03 and (flags == "TR" or ((flags == "R" or flags == "DR") and particle)):
        assert np.mean(result["directionOut"][refl],axis=0) == pytest.approx(np.average(r[mask_r],weights=R[mask_r],axis=0),rel=err)
    if R_mean_r > 0.03 and (flags == "R" or flags == "DR") and not particle:
        assert np.mean(result["directionOut"][refl],axis=0) == pytest.approx(np.average(r[mask_r],axis=0),rel=err)
        assert np.std(result["directionOut"][refl],axis=0) == pytest.approx(np.std(r[mask_r],axis=0),rel=err)

    assert np.all(result["mediumIdxOut"][trans] == VACUUM_IDX)
    assert np.all(result["mediumIdxOut"][refl] == waterIdx)

    if flags == "R" or flags == "DR":
        assert trans.sum() == 0
        if particle and (1 - R_mean_r) > 0.01:
            assert absorbed.sum() > 0
        elif R_mean_r > 0.01:
            cO = result["contribOut"]
            assert np.mean(cO[~absorbed]) == pytest.approx(R_mean_r * np.mean(c[~absorbed]),rel=err)
    if flags == "T":
        assert refl.sum() == 0
        if particle and R_mean_t > 0.01:
             absorbed.sum() > 0
        if not particle and (1 - R_mean_t) > 0.01:
            cO = result["contribOut"]
            assert np.mean(cO[~absorbed]) == pytest.approx((1-R_mean_t) * np.mean(c[~absorbed]),rel=err)
        if not particle and not camera and (1 - R_mean_t) > 0.01:
            cH = result["contribHit"]
            assert np.mean(cH[~absorbed]) == pytest.approx((1-R_mean_t) * np.mean(c[~absorbed]),rel=err)
    if flags == "TR":
        if R_mean_r > 0.01:
            assert refl.sum() > 0
        if (1 - R_mean_t) > 0.01:
            assert trans.sum() > 0
        assert absorbed.sum() == 0
        if particle:
            # check particle is not both reflected and detected
            np.all((result["hitSuccess"] == 0) == refl)
        else:
            assert np.allclose(result["contribOut"], c)
