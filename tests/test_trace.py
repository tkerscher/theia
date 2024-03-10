import pytest

import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl

import theia.estimator
import theia.light
import theia.material
import theia.random
import theia.scene
import theia.trace
import theia.units as u

from common.models import WaterModel


@pytest.mark.parametrize("disableDirect", [True, False])
@pytest.mark.parametrize("disableTarget", [True, False])
@pytest.mark.parametrize("limitTime", [True, False])
def test_VolumeTracer(disableDirect: bool, disableTarget: bool, limitTime: bool):
    N = 32 * 256
    N_LAMBDA = 4
    N_SCATTER = 6
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 1.0 * u.us if limitTime else 100.0 * u.us
    light_pos = (-1.0, -7.0, 0.0) * u.m
    light_intensity = 1000.0
    target_pos, target_radius = (5.0, 2.0, -8.0) * u.m, 4.0 * u.m

    # create water medium
    water = WaterModel().createMedium()
    store = theia.material.MaterialStore([], media=[water])

    # estimate max speed
    d_min = (
        np.sqrt(np.square(np.subtract(light_pos, target_pos)).sum(-1)) - target_radius
    )
    v_max = np.max(water.group_velocity)
    t_min = d_min / v_max + T0

    # create pipeline
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    rays = theia.light.SphericalRaySource(position=light_pos)
    photons = theia.light.UniformPhotonSource(
        intensity=light_intensity,
        timeRange=(T0, T1),
    )
    source = theia.light.ModularLightSource(rays, photons, N_LAMBDA)
    recorder = theia.estimator.HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.VolumeTracer(
        N,
        source,
        recorder,
        rng,
        medium=store.media["water"],
        nScattering=N_SCATTER,
        callback=stats,
        maxTime=T_MAX,
        target=theia.scene.SphereBBox(target_pos, target_radius),
        disableDirectLighting=disableDirect,
        disableTargetSampling=disableTarget,
    )
    # run pipeline
    pl.runPipeline([rng, rays, photons, source, tracer, recorder])

    # check hits
    hits = recorder.view(0)
    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]

    assert np.allclose(np.square(hits["position"]).sum(-1), target_radius**2)
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX

    # check config via stats
    assert stats.created == tracer.batchSize
    assert stats.scattered > 0
    if disableDirect and disableTarget:
        assert stats.absorbed > 0
        assert stats.detected > 0
        if not limitTime:
            assert len(hits) == stats.detected * N_LAMBDA
    elif disableDirect and not disableTarget:
        assert stats.absorbed > 0
        assert stats.detected == 0
        if not limitTime:
            assert len(hits) > stats.scattered * N_LAMBDA
    elif not disableDirect and disableTarget:
        assert stats.absorbed == 0
        assert stats.detected > 0
        if not limitTime:
            assert len(hits) == stats.detected * N_LAMBDA
    elif not disableDirect and not disableTarget:
        assert stats.absorbed > 0
        assert stats.detected > 0
        if not limitTime:
            assert len(hits) > (stats.detected + stats.scattered) * N_LAMBDA

    # TODO: more sophisticated tests...


@pytest.mark.parametrize("disableDirect", [True, False])
@pytest.mark.parametrize("disableVolumeBorder", [True, False])
@pytest.mark.parametrize("disableTransmission", [True, False])
@pytest.mark.parametrize("disableTarget", [True, False])
def test_SceneTracer(
    disableDirect: bool,
    disableVolumeBorder: bool,
    disableTransmission: bool,
    disableTarget: bool,
):
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 256
    N_LAMBDA = 4
    MAX_PATH = 10
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 1.0 * u.us
    light_pos = (-1.0, -7.0, 0.0) * u.m
    light_intensity = 1000.0

    # create materials
    water = WaterModel().createMedium()
    glass = theia.material.BK7Model().createMedium()
    mat = theia.material.Material("mat", glass, water, flags=("DR", "B"))
    matStore = theia.material.MaterialStore([mat])
    # create scene
    store = theia.scene.MeshStore(
        {"cube": "assets/cone.stl", "sphere": "assets/sphere.stl"}
    )
    r, d = 40.0 * u.m, 5.0 * u.m
    r_scale = 0.99547149974733 * u.m  # radius of inscribed sphere (icosphere)
    r_insc = r * r_scale
    x, y, z = 10.0, 5.0, -5.0
    t1 = theia.scene.Transform.Scale(r, r, r).translate(x, y, z + r + d)
    c1 = store.createInstance("sphere", "mat", transform=t1, detectorId=0)
    t2 = theia.scene.Transform.Scale(r, r, r).translate(x, y, z - r - d)
    c2 = store.createInstance("sphere", "mat", transform=t2, detectorId=1)
    targets = [
        theia.scene.SphereBBox((x, y, z + r + d), r),
        theia.scene.SphereBBox((x, y, z - r - d), r),
    ]
    scene = theia.scene.Scene(
        [c1, c2], matStore.material, medium=matStore.media["water"], targets=targets
    )

    # calculate min time
    target_pos = (x, y, z - r - d) * u.m  # detector #1
    d_min = np.sqrt(np.square(np.subtract(target_pos, light_pos)).sum(-1)) - r
    v_max = np.max(water.group_velocity)
    t_min = d_min / v_max + T0

    # create pipeline stages
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    rays = theia.light.SphericalRaySource(position=light_pos)
    photons = theia.light.UniformPhotonSource(
        intensity=light_intensity,
        timeRange=(T0, T1),
    )
    source = theia.light.ModularLightSource(rays, photons, N_LAMBDA)
    recorder = theia.estimator.HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneTracer(
        N,
        source,
        recorder,
        rng,
        scene,
        maxPathLength=MAX_PATH,
        targetIdx=1,
        maxTime=T_MAX,
        callback=stats,
        disableDirectLighting=disableDirect,
        disableTargetSampling=disableTarget,
        disableTransmission=disableTransmission,
        disableVolumeBorder=disableVolumeBorder,
    )
    # run pipeline
    pl.runPipeline([rng, rays, photons, source, tracer, recorder])

    # check hits
    hits = recorder.view(0)
    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]

    r_hits = np.sqrt(np.square(hits["position"]).sum(-1))
    assert np.all((r_hits <= 1.0) & (r_hits >= r_scale))
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX

    # TODO: more sophisticated tests


def test_EventStatisticCallback():
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 1024
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns

    # create materials
    water = WaterModel().createMedium()
    glass = theia.material.BK7Model().createMedium()
    mat = theia.material.Material("mat", glass, water, flags=("DR", "B"))
    vol = theia.material.Material("vol", glass, water, flags="V")
    absorber = theia.material.Material("abs", glass, water, flags="B")
    matStore = theia.material.MaterialStore([mat, vol, absorber])
    # create scene
    store = theia.scene.MeshStore({"sphere": "assets/sphere.stl"})
    r, d = 20.0, 5.0
    t1 = theia.scene.Transform.Scale(r, r, r).translate(17.0, 17.0, 17.0)
    c1 = store.createInstance("sphere", "mat", transform=t1, detectorId=0)
    t2 = theia.scene.Transform.Scale(r, r, r).translate(-17.0, -17.0, -17.0)
    c2 = store.createInstance("sphere", "mat", transform=t2, detectorId=1)
    t3 = theia.scene.Transform.Scale(r, r, r).translate(17.0, 17.0, -17.0)
    c3 = store.createInstance("sphere", "vol", transform=t3)
    t4 = theia.scene.Transform.Scale(r, r, r).translate(-17.0, 17.0, 17.0)
    c4 = store.createInstance("sphere", "abs", transform=t4)
    bbox = theia.scene.RectBBox((-50.0,) * 3, (50.0,) * 3)
    scene = theia.scene.Scene(
        [c1, c2, c3, c4], matStore.material, medium=matStore.media["water"], bbox=bbox
    )

    # create pipeline
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    rays = theia.light.SphericalRaySource()
    photons = theia.light.UniformPhotonSource(timeRange=(T0, T1))
    source = theia.light.ModularLightSource(rays, photons, 1)
    response = theia.estimator.EmptyResponse()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneTracer(
        N,
        source,
        response,
        rng,
        callback=stats,
        scene=scene,
        targetIdx=1,
        maxPathLength=1,
        scatterCoefficient=0.005 / u.m,
        maxTime=100.0 * u.ns,
        disableTargetSampling=True,
    )
    # assert stats start empty
    assert stats.created == 0
    assert stats.absorbed == 0
    assert stats.hit == 0
    assert stats.detected == 0
    assert stats.scattered == 0
    assert stats.lost == 0
    assert stats.decayed == 0
    assert stats.volume == 0
    assert stats.error == 0
    # run pipeline
    pl.runPipeline([rng, rays, photons, source, tracer])

    # check stats again
    assert stats.created == N
    assert stats.absorbed > 0
    assert stats.hit > 0
    assert stats.detected > 0
    assert stats.scattered > 0
    assert stats.lost > 0
    assert stats.decayed > 0
    assert stats.volume > 0
    assert stats.error == 0
    total = (
        stats.absorbed
        + stats.hit
        + stats.detected
        + stats.scattered
        + stats.lost
        + stats.decayed
        + stats.volume
    )
    assert total == N
    # check if we can reset it
    stats.reset()
    assert stats.created == 0
    assert stats.absorbed == 0
    assert stats.hit == 0
    assert stats.detected == 0
    assert stats.scattered == 0
    assert stats.lost == 0
    assert stats.decayed == 0
    assert stats.volume == 0
    assert stats.error == 0


def test_TrackRecordCallback():
    N = 32 * 256
    N_LAMBDA = 4
    N_SCATTER = 6
    LENGTH = N_SCATTER + 2  # one more than needed
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 500.0 * u.ns
    target_pos, target_radius = (5.0, 2.0, -8.0) * u.m, 4.0 * u.m
    light_pos = (100.0, -50.0, 20.0) * u.m
    light_dir = (1.0, 0.0, 0.0) * u.m

    # create water medium
    water = WaterModel().createMedium()
    store = theia.material.MaterialStore([], media=[water])

    # create pipeline
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    rays = theia.light.PencilRaySource(position=light_pos, direction=light_dir)
    photons = theia.light.UniformPhotonSource(timeRange=(T0, T1))
    source = theia.light.ModularLightSource(rays, photons, N_LAMBDA)
    response = theia.estimator.EmptyResponse()
    track = theia.trace.TrackRecordCallback(N, LENGTH)
    tracer = theia.trace.VolumeTracer(
        N,
        source,
        response,
        rng,
        callback=track,
        medium=store.media["water"],
        nScattering=N_SCATTER,
        scatterCoefficient=0.1,
        maxTime=T_MAX,
        traceBBox=theia.scene.RectBBox((-200.0 * u.m,) * 3, (200.0 * u.m,) * 3),
        target=theia.scene.SphereBBox(target_pos, target_radius),
    )
    # run pipeline
    pl.runPipeline([rng, rays, photons, source, tracer, track])

    # check result
    tracks, lengths, codes = track.result(0)
    assert tracks.shape == (N, LENGTH, 4)
    assert np.all(tracks[:, 0, :3] == light_pos)
    assert tracks[:, 0, 3].min() >= T0 and tracks[:, 0, 3].max() <= T1
    # since we use a pencil beam in x direction, y and z for the first scatter
    # are still deterministic
    assert np.all(tracks[:, 1, 1] == light_pos[1])
    assert np.all(tracks[:, 1, 2] == light_pos[2])
    # check length is in range
    assert lengths.min() >= 1 and lengths.max() <= N_SCATTER
    # check result codes in range
    minCode = min(code.value for code in theia.trace.EventResultCode)
    maxCode = max(code.value for code in theia.trace.EventResultCode)
    assert codes.min() >= minCode and codes.max() <= maxCode


def test_volumeBorder():
    """
    Simple test we handle volume borders correctly, i.e.:
    - no refraction
    - update ray parameters

    Test by shooting laser from air to glass and marking the glass as volume
    boundary.
    """
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 256
    LAMBDA = 500.0 * u.nm

    # create materials
    model = theia.material.BK7Model()
    glass = model.createMedium()
    mat = theia.material.Material("mat", glass, None, flags=("V", "B"))
    matStore = theia.material.MaterialStore([mat])
    # create scene
    store = theia.scene.MeshStore({"cube": "assets/cube.ply"})
    trafo = theia.scene.Transform.Scale(50.0, 50.0, 50.0).translate(75.0, 0.0, 0.0)
    cube = store.createInstance("cube", "mat", transform=trafo)
    scene = theia.scene.Scene([cube], matStore.material)

    # create scene
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    rays = theia.light.PencilRaySource(  # TODO: would cone source work better?
        position=(0.0, 0.0, 0.0),
        # important: hit cube not straight on but in angle to test for no refraction
        direction=(0.8, 0.36, 0.48),
    )
    photons = theia.light.UniformPhotonSource(
        lambdaRange=(LAMBDA, LAMBDA),  # const lambda
        timeRange=(0.0, 0.0),  # const time
        intensity=1.0,
    )
    source = theia.light.ModularLightSource(rays, photons, 1)
    estimator = theia.estimator.EmptyResponse()
    tracker = theia.trace.TrackRecordCallback(N, 4)
    tracer = theia.trace.SceneTracer(
        N,
        source,
        estimator,
        rng,
        callback=tracker,
        scene=scene,
        targetIdx=1,
        maxPathLength=2,
        disableTargetSampling=True,  # there's no scattering anyway
    )
    # run pipeline
    pl.runPipeline([rng, rays, photons, source, tracer, tracker])

    # retrieve result
    track, lengths, codes = tracker.result(0)
    # check for no refraction (straight line)
    assert lengths.min() >= 2
    d1 = track[:, 1, :3] - track[:, 0, :3]
    l1 = np.sqrt(np.square(d1).sum(-1))
    d2 = track[:, 2, :3] - track[:, 1, :3]
    l2 = np.sqrt(np.square(d2).sum(-1))
    cos_theta = np.multiply(d1, d2).sum(-1) / (l1 * l2)
    assert cos_theta.min() >= 1.0 - 1e-5
    # check ray slowed down
    t1 = track[:, 1, 3] - track[:, 0, 3]
    t2 = track[:, 2, 3] - track[:, 1, 3]
    n1 = theia.material.speed_of_light / (l1 / t1)
    n2 = theia.material.speed_of_light / (l2 / t2)
    n2_exp = theia.material.speed_of_light / model.group_velocity(LAMBDA)
    assert np.abs(n1 - 1.0).max() < 5e-5
    assert np.abs(n2 - n2_exp).max() < 5e-5


@pytest.mark.parametrize(
    "flag,reflectance,err", [("T", 0.0, 0.0), ("R", 1.0, 0.0), ("TR", 0.0516, 0.005)]
)
def test_tracer_reflection(flag, reflectance, err):
    """
    We simulate a very simply laser setup to check if the transmission/reflection
    code is working properly:

                                __                          _________
    ------+                     \ \       transmit         |
    LASER | - - - - - - - - - - -\ \ - - - - - - - - - - - | TARGET 2
    ------+                      |\_\                      |_________
                                 |
                                 |  reflect                A y
                            _____|______                   |
                           |  TARGET 1  |                  |        x
                                                          -+-------->

    (0.0,0.0) is at the center of the splitter.
    """
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 256

    # create materials
    glass = theia.material.BK7Model().createMedium()
    # only enable transmission on rays going outwards the splitter
    mat = theia.material.Material("mat", glass, None, flags=(flag, "T"))
    det = theia.material.Material("det", glass, None, flags="DB")
    matStore = theia.material.MaterialStore([mat, det])
    # create scene
    store = theia.scene.MeshStore({"cube": "assets/cube.ply"})
    splitter_trans = (
        theia.scene.Transform()
        .Scale(0.1, 5.0, 5.0)
        .rotate(0.0, 0.0, 1.0, np.radians(45.0))
        .translate(0.141, 0.0, 0.0)
    )
    splitter = store.createInstance("cube", "mat", transform=splitter_trans)
    ref_trans = theia.scene.Transform.Scale(50.0, 0.5, 50.0).translate(0.0, -50.0, 0.0)
    ref_det = store.createInstance("cube", "det", transform=ref_trans, detectorId=1)
    trans_trans = theia.scene.Transform.Scale(0.5, 50.0, 50.0).translate(50.0, 0.0, 0.0)
    trans_det = store.createInstance("cube", "det", transform=trans_trans, detectorId=2)
    scene = theia.scene.Scene([splitter, trans_det, ref_det], matStore.material)

    # create pipeline
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    rays = theia.light.PencilRaySource(
        position=(-20.0, 0.0, 0.0) * u.m, direction=(1.0, 0.0, 0.0)
    )
    photons = theia.light.UniformPhotonSource(
        lambdaRange=(500.0, 500.0) * u.nm,  # const lambda
        timeRange=(0.0, 0.0),  # const time
        intensity=1.0,
    )
    source = theia.light.ModularLightSource(rays, photons, 1)
    recorder = theia.estimator.HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneTracer(
        N,
        source,
        recorder,
        rng,
        scene=scene,
        targetIdx=1,
        maxPathLength=4,
        callback=stats,
        disableTargetSampling=True,  # there's no scattering anyway
    )
    pipeline = pl.Pipeline([rng, rays, photons, source, tracer, recorder])
    # run pipeline
    pipeline.run(0)
    hits_ref = recorder.view(0)
    # run for second detector
    tracer.setParam("targetIdx", 2)
    pipeline.run(1)
    hits_trans = recorder.view(1)

    # check result
    assert hits_ref.count + hits_trans.count == N
    reflectance_sim = hits_ref.count / N
    assert np.abs(reflectance_sim - reflectance) <= err
