import pytest

import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl

import theia.camera
import theia.response
import theia.light
import theia.material
import theia.random
import theia.scene
import theia.trace
import theia.units as u

from theia.scene import Transform
from theia.target import SphereTarget, SphereTargetGuide
from common.models import WaterModel


@pytest.mark.parametrize("disableDirect", [True, False])
@pytest.mark.parametrize("disableTarget", [True, False])
@pytest.mark.parametrize("limitTime", [True, False])
@pytest.mark.parametrize("polarized", [True, False])
def test_VolumeForwardTracer(
    disableDirect: bool, disableTarget: bool, limitTime: bool, polarized: bool
):
    N = 32 * 256
    N_SCATTER = 6
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 1.0 * u.us if limitTime else 100.0 * u.us
    light_pos = (-1.0, -7.0, 0.0) * u.m
    light_budget = 1000.0
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
    photons = theia.light.UniformWavelengthSource()
    source = theia.light.SphericalLightSource(
        position=light_pos, budget=light_budget, timeRange=(T0, T1)
    )
    target = SphereTarget(position=target_pos, radius=target_radius)
    recorder = theia.response.HitRecorder(polarized=polarized)
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.VolumeForwardTracer(
        N,
        source,
        target,
        photons,
        recorder,
        rng,
        medium=store.media["water"],
        nScattering=N_SCATTER,
        callback=stats,
        # callback=track,
        maxTime=T_MAX,
        polarized=polarized,
        disableDirectLighting=disableDirect,
        disableTargetSampling=disableTarget,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check hits
    hits = recorder.queue.view(0)
    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]

    assert np.allclose(np.square(hits["position"]).sum(-1), 1.0)
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX

    # sanity check polarization
    if polarized:
        # check stokes is normalized
        assert np.abs(hits["stokes"][:, 0] - 1.0).max() < 1e-6
        assert np.square(hits["stokes"][:, 1:]).sum(-1).max() <= 1.0
        assert hits["stokes"][:, 1:].max() <= 1.0
        assert hits["stokes"][:, 1:].min() >= -1.0
        # check polarization ref is perpendicular to plane of incidence and normalized
        polRef = hits["polarizationRef"]
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        polRef_exp = np.cross(hits["direction"], hits["normal"])
        d = np.square(polRef_exp).sum(-1)
        mask = d > 1e-7
        polRef_exp /= np.sqrt(d)[:, None]
        # for very small angle we reuse the old polRef to minimize error
        assert np.abs(np.abs((polRef_exp * polRef).sum(-1))[mask] - 1.0).max() < 1e-6

    # check config via stats
    # NOTE: We no longer report hits with zero contrib.
    #       The asserts commented out no longer hold
    assert stats.created == tracer.batchSize
    assert stats.scattered > 0
    if disableDirect and disableTarget:
        assert stats.absorbed > 0
        assert stats.detected > 0
        # if not limitTime:
        #     assert len(hits) == stats.detected
    elif disableDirect and not disableTarget:
        assert stats.absorbed > 0
        assert stats.detected == 0
        # if not limitTime:
        #     assert len(hits) > stats.scattered
    elif not disableDirect and disableTarget:
        assert stats.absorbed == 0
        assert stats.detected > 0
        # if not limitTime:
        #     assert len(hits) == stats.detected
    elif not disableDirect and not disableTarget:
        assert stats.absorbed > 0
        assert stats.detected == 0
        # if not limitTime:
        #     assert len(hits) > (stats.detected + stats.scattered)

    # TODO: more sophisticated tests...


@pytest.mark.parametrize("disableDirect", [True, False])
@pytest.mark.parametrize("disableTarget", [True, False])
@pytest.mark.parametrize("limitTime", [True, False])
@pytest.mark.parametrize("polarized", [True, False])
def test_VolumeBackwardTracer(
    disableDirect: bool, disableTarget: bool, limitTime: bool, polarized: bool
):
    N = 32 * 256
    N_SCATTER = 6
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 1.0 * u.us
    light_pos = (-1.0, -7.0, 0.0) * u.m
    light_budget = 1000.0
    target_pos, target_radius = (5.0, 2.0, -8.0) * u.m, 4.0 * u.m
    target = SphereTarget(position=target_pos, radius=target_radius)

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
    photons = theia.light.UniformWavelengthSource()
    light = theia.light.SphericalLightSource(
        position=light_pos,
        budget=light_budget,
        timeRange=(T0, T1),
    )
    camera = theia.camera.SphereCamera(position=target_pos, radius=target_radius)
    recorder = theia.response.HitRecorder(polarized=polarized)
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.VolumeBackwardTracer(
        N,
        light,
        camera,
        photons,
        recorder,
        rng,
        callback=stats,
        medium=store.media["water"],
        nScattering=N_SCATTER,
        target=None if disableTarget else target,
        maxTime=T_MAX if limitTime else 10.0 * u.ms,
        polarized=polarized,
        disableDirectLighting=disableDirect,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check hits
    hits = recorder.queue.view(0)
    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]

    assert np.allclose(np.square(hits["position"]).sum(-1), 1.0)
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.min(hits["time"]) >= t_min
    if limitTime:
        assert np.max(hits["time"]) <= T_MAX
    else:
        assert np.max(hits["time"]) > T_MAX

    # sanity check polarization
    if polarized:
        # check stokes is normalized
        assert np.abs(hits["stokes"][:, 0] - 1.0).max() < 1e-6
        assert np.square(hits["stokes"][:, 1:]).sum(-1).max() <= 1.0
        assert hits["stokes"][:, 1:].max() <= 1.0
        assert hits["stokes"][:, 1:].min() >= -1.0
        # check polarization ref is perpendicular to plane of incidence and normalized
        polRef = hits["polarizationRef"]
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        polRef_exp = np.cross(hits["direction"], hits["normal"])
        d = np.square(polRef_exp).sum(-1)
        mask = d > 1e-7
        polRef_exp /= np.sqrt(d)[:, None]
        # for very small angle we reuse the old polRef to minimize error
        assert np.abs(np.abs((polRef_exp * polRef).sum(-1))[mask] - 1.0).max() < 1e-6

    # check config via stats
    assert stats.scattered > 0
    if disableTarget:
        assert stats.absorbed == 0
    else:
        assert stats.absorbed > 0
    if disableDirect:
        assert stats.created == tracer.batchSize
        # there will only be hits from shadow rays
        assert stats.detected == 0
    else:
        assert stats.created == 2 * tracer.batchSize
        assert stats.detected > 0


@pytest.mark.parametrize("disableDirect", [True, False])
@pytest.mark.parametrize("disableVolumeBorder", [True, False])
@pytest.mark.parametrize("disableTransmission", [True, False])
@pytest.mark.parametrize("disableTarget", [True, False])
@pytest.mark.parametrize("polarized", [True, False])
def test_SceneForwardTracer(
    polarized: bool,
    disableDirect: bool,
    disableVolumeBorder: bool,
    disableTransmission: bool,
    disableTarget: bool,
):
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 256
    MAX_PATH = 10
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 1.0 * u.us
    light_pos = (-1.0, -7.0, 0.0) * u.m
    light_budget = 1000.0

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
    t1 = Transform.TRS(scale=r, translate=(x, y, z + r + d))
    c1 = store.createInstance("sphere", "mat", t1, detectorId=0)
    t2 = Transform.TRS(
        scale=r, translate=(x, y, z - r - d), rotate=(0.0, 0.0, 1.0, 110.0)
    )
    c2 = store.createInstance("sphere", "mat", t2, detectorId=1)
    scene = theia.scene.Scene(
        [c1, c2], matStore.material, medium=matStore.media["water"]
    )
    guide = SphereTargetGuide(position=(x, y, z - r - d), radius=r)

    # calculate min time
    target_pos = (x, y, z - r - d) * u.m  # detector #1
    d_min = np.sqrt(np.square(np.subtract(target_pos, light_pos)).sum(-1)) - r
    v_max = np.max(water.group_velocity)
    t_min = d_min / v_max + T0

    # create pipeline stages
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    photons = theia.light.UniformWavelengthSource()
    source = theia.light.SphericalLightSource(
        position=light_pos, timeRange=(T0, T1), budget=light_budget
    )
    recorder = theia.response.HitRecorder(polarized=polarized)
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneForwardTracer(
        N,
        source,
        photons,
        recorder,
        rng,
        scene,
        maxPathLength=MAX_PATH,
        targetIdx=1,
        targetGuide=None if disableTarget else guide,
        maxTime=T_MAX,
        polarized=polarized,
        callback=stats,
        disableDirectLighting=disableDirect,
        disableTransmission=disableTransmission,
        disableVolumeBorder=disableVolumeBorder,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check hits
    hits = recorder.queue.view(0)
    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]

    r_hits = np.sqrt(np.square(hits["position"]).sum(-1))
    assert np.all((r_hits <= 1.0) & (r_hits >= r_scale))
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX

    # sanity check polarization
    if polarized:
        # check stokes is normalized
        assert np.abs(hits["stokes"][:, 0] - 1.0).max() < 1e-6
        assert np.sqrt(np.square(hits["stokes"][:, 1:]).sum(-1).max()) - 1.0 <= 1e-6
        assert hits["stokes"][:, 1:].max() <= 1.0
        assert hits["stokes"][:, 1:].min() >= -1.0
        # check polarization ref is perpendicular to plane of incidence and normalized
        # note: this only works because we use spheres and only scale and translate them
        polRef = hits["polarizationRef"]
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        polRef_exp = np.cross(hits["direction"], hits["normal"])
        d = np.square(polRef_exp).sum(-1)
        mask = d > 1e-7
        polRef_exp /= np.sqrt(d)[:, None]
        # for very small angle we reuse the old polRef to minimize error
        assert np.abs(np.abs((polRef_exp * polRef).sum(-1))[mask] - 1.0).max() < 1e-6

    # TODO: more sophisticated tests


@pytest.mark.parametrize("disableDirect", [True, False])
@pytest.mark.parametrize("disableVolumeBorder", [True, False])
@pytest.mark.parametrize("disableTransmission", [True, False])
@pytest.mark.parametrize("polarized", [True, False])
def test_SceneBackwardTracer(
    polarized: bool,
    disableDirect: bool,
    disableVolumeBorder: bool,
    disableTransmission: bool,
):
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 256
    N_SCATTER = 6
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 100000.0 * u.us
    light_pos = (-1.0, -7.0, 0.0) * u.m
    light_budget = 1000.0

    # create materials
    water = WaterModel().createMedium()
    glass = theia.material.BK7Model().createMedium()
    mat = theia.material.Material("mat", glass, water, flags=("DR", "B"))
    matAbs = theia.material.Material("abs", None, water, flags="B")
    matVol = theia.material.Material("vol", water, water, flags="V")
    matStore = theia.material.MaterialStore([mat, matAbs, matVol])
    # create scene
    store = theia.scene.MeshStore(
        {"cube": "assets/cube.ply", "sphere": "assets/sphere.stl"}
    )
    r, d = 40.0 * u.m, 5.0 * u.m
    r_scale = 0.99547149974733 * u.m  # radius of inscribed sphere (icosphere)
    r_insc = r * r_scale
    x, y, z = 10.0, 5.0, -5.0
    t1 = Transform.TRS(scale=r, translate=(x, y, z + r + d))
    c1 = store.createInstance("sphere", "mat", t1, detectorId=0)
    t2 = Transform.TRS(scale=r, translate=(x, y, z - r - d))
    c2 = store.createInstance("sphere", "mat", t2, detectorId=1)
    t3 = Transform.TRS(scale=r, translate=(-x, -y, z + r + d))
    c3 = store.createInstance("sphere", "abs", t3)
    instances = [c1, c2, c3]
    t4 = Transform.TRS(scale=r, translate=(-x, -y, z - r - d))
    c4 = store.createInstance("sphere", "vol", t4)
    if not disableVolumeBorder:
        instances.append(c4)
    scene = theia.scene.Scene(
        instances, matStore.material, medium=matStore.media["water"]
    )

    # calculate min time
    target_pos = (x, y, z - r - d) * u.m  # detector #1
    d_min = np.sqrt(np.square(np.subtract(target_pos, light_pos)).sum(-1)) - r
    v_max = np.max(water.group_velocity)
    t_min = d_min / v_max + T0

    # create pipeline stages
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    photons = theia.light.UniformWavelengthSource()
    source = theia.light.SphericalLightSource(
        position=light_pos,
        timeRange=(T0, T1),
        budget=light_budget,
    )
    camera = theia.camera.SphereCamera(
        position=target_pos,
        radius=r,
    )
    recorder = theia.response.HitRecorder(polarized=polarized)
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneBackwardTracer(
        N,
        source,
        camera,
        photons,
        recorder,
        rng,
        scene,
        callback=stats,
        maxPathLength=N_SCATTER,
        maxTime=T_MAX,
        polarized=polarized,
        disableDirectLighting=disableDirect,
        disableTransmission=disableTransmission,
        disableVolumeBorder=disableVolumeBorder,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check hits
    hits = recorder.queue.view(0)
    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]

    # sanity check polarization
    if polarized:
        # check stokes is normalized
        assert np.abs(hits["stokes"][:, 0] - 1.0).max() < 1e-6
        assert np.sqrt(np.square(hits["stokes"][:, 1:]).sum(-1).max()) - 1.0 <= 1e-6
        assert hits["stokes"][:, 1:].max() <= 1.0
        assert hits["stokes"][:, 1:].min() >= -1.0
        # check polarization ref is perpendicular to plane of incidence and normalized
        # note: this only works because we use spheres and only scale and translate them
        polRef = hits["polarizationRef"]
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        polRef_exp = np.cross(hits["direction"], hits["normal"])
        d = np.square(polRef_exp).sum(-1)
        mask = d > 1e-7
        polRef_exp /= np.sqrt(d)[:, None]
        # for very small angle we reuse the old polRef to minimize error
        assert np.abs(np.abs((polRef_exp * polRef).sum(-1))[mask] - 1.0).max() < 1e-6

    # check config via stats
    assert stats.scattered > 0
    assert stats.absorbed > 0
    if disableDirect:
        assert stats.created == tracer.batchSize
        # there will only be hits from shadow rays
        assert stats.detected == 0
    else:
        assert stats.created == 2 * tracer.batchSize
        assert stats.detected > 0
    if disableVolumeBorder:
        assert stats.volume == 0
    else:
        assert stats.volume > 0


@pytest.mark.parametrize("disableVolumeBorder", [True, False])
@pytest.mark.parametrize("disableTransmission", [True, False])
@pytest.mark.parametrize("disableTarget", [True, False])
def test_SceneBackwardTargetTracer(
    disableVolumeBorder: bool, disableTransmission: bool, disableTarget: bool
) -> None:
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing not supported")

    N = 32 * 256
    MAX_PATH = 10
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 1.0 * u.us
    cam_pos = (-1.0, -7.0, 0.0) * u.m

    # create materials
    water = WaterModel().createMedium()
    glass = theia.material.BK7Model().createMedium()
    mat = theia.material.Material("mat", glass, water, flags=("RL", "B"))
    matStore = theia.material.MaterialStore([mat])
    # create scene
    store = theia.scene.MeshStore(
        {"cube": "assets/cube.ply", "sphere": "assets/sphere.stl"}
    )
    r, d = 40.0 * u.m, 5.0 * u.m
    r_scale = 0.99547149974733 * u.m  # radius of inscribed sphere (icosphere)
    r_insc = r * r_scale
    x, y, z = 10.0, 5.0, -5.0
    t1 = Transform.TRS(scale=r, translate=(x, y, z + r + d))
    c1 = store.createInstance("sphere", "mat", t1, detectorId=1)
    t2 = Transform.TRS(
        scale=r, translate=(x, y, z - r - d), rotate=(0.0, 0.0, 1.0, 110.0)
    )
    c2 = store.createInstance("sphere", "mat", t2, detectorId=2)
    scene = theia.scene.Scene(
        [c1, c2], matStore.material, medium=matStore.media["water"]
    )
    guide = SphereTargetGuide(position=(x, y, z - r - d), radius=r)

    # calculate min time
    target_pos = (x, y, z - r - d) * u.m  # detector #2
    d_min = np.sqrt(np.square(np.subtract(target_pos, cam_pos)).sum(-1)) - r
    v_max = np.max(water.group_velocity)
    t_min = d_min / v_max + T0

    # create pipeline stages
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    photons = theia.light.UniformWavelengthSource()
    camera = theia.camera.PointCamera(position=cam_pos, timeDelta=T0)
    recorder = theia.response.HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneBackwardTargetTracer(
        N,
        camera,
        photons,
        recorder,
        rng,
        scene,
        maxPathLength=MAX_PATH,
        targetId=2,
        targetGuide=None if disableTarget else guide,
        maxTime=T_MAX,
        callback=stats,
        disableTransmission=disableTransmission,
        disableVolumeBorder=disableVolumeBorder,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check hits
    hits = recorder.queue.view(0)
    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]

    r_hits = np.sqrt(np.square(hits["position"]).sum(-1))
    assert np.all((r_hits <= 1.0) & (r_hits >= r_scale))
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX
    assert np.all(hits["contrib"] >= 0.0)


@pytest.mark.parametrize("polarized", [True, False])
@pytest.mark.parametrize("disableTransmission", [True, False])
def test_BidirectionalPathTracer(
    polarized: bool,
    disableTransmission: bool,
):
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 64
    T0, T1 = 10 * u.ns, 20.0 * u.ns
    T_MAX = 100.0 * u.us
    L_LIGHT = 8
    L_CAMERA = 8

    det_pos = (-5.0, 0.0, 0.0) * u.m
    src_pos = (5.0, 0.0, 0.0) * u.m
    r_inner = 0.95
    r_outer = 1.0
    shield_scale = 80.0
    shield_size = (0.1, shield_scale, shield_scale)  # scale
    bbox = ((-500.0 * u.m,) * 3, (500.0 * u.m,) * 3)

    # create materials
    water = WaterModel().createMedium()
    glass = theia.material.BK7Model().createMedium()
    sphere_inner = theia.material.Material("sph_inner", None, glass, flags="T")
    sphere_outer = theia.material.Material("sph_outer", glass, water, flags="T")
    absorber = theia.material.Material("absorber", None, water, flags="B")
    matStore = theia.material.MaterialStore([sphere_inner, sphere_outer, absorber])
    # create scene
    store = theia.scene.MeshStore(
        {"cube": "assets/cube.ply", "sphere": "assets/sphere.stl"}
    )
    t_det_inner = Transform.TRS(scale=r_inner, translate=det_pos)
    det_inner = store.createInstance("sphere", "sph_inner", t_det_inner)
    t_det_outer = Transform.TRS(scale=r_outer, translate=det_pos)
    det_outer = store.createInstance("sphere", "sph_outer", t_det_outer)
    t_src_inner = Transform.TRS(scale=r_inner, translate=src_pos)
    src_inner = store.createInstance("sphere", "sph_inner", t_src_inner)
    t_src_outer = Transform.TRS(scale=r_outer, translate=src_pos)
    src_outer = store.createInstance("sphere", "sph_outer", t_src_outer)
    t_shield = Transform.TRS(scale=shield_size)
    shield = store.createInstance("cube", "absorber", t_shield)
    scene = theia.scene.Scene(
        [det_outer, det_inner, src_outer, src_inner, shield],
        matStore.material,
        bbox=theia.scene.RectBBox(*bbox),
    )

    # calc expected shortest time for light paths
    d_min = 2.0 * np.sqrt(det_pos[0] ** 2 + shield_scale**2)
    v_max = np.max(water.group_velocity)
    t_min = d_min / v_max + T0  # short path in air/glass ignored (should be fine)

    # create pipeline
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    cam = theia.camera.PencilCamera(rayPosition=src_pos)
    # cam = theia.camera.FlatCamera(transform=det_trafo)
    ph = theia.light.UniformWavelengthSource()
    src = theia.light.PencilLightSource(position=det_pos, timeRange=(T0, T1))
    rec = theia.response.HitRecorder(polarized=polarized)
    # stats = theia.trace.EventStatisticCallback()
    track = theia.trace.TrackRecordCallback(N, L_CAMERA + 2, polarized=polarized)
    trace = theia.trace.BidirectionalPathTracer(
        N,
        src,
        cam,
        ph,
        rec,
        rng,
        scene,
        # callback=stats,
        callback=track,
        callbackScope="camera",
        lightPathLength=L_LIGHT,
        cameraPathLength=L_CAMERA,
        maxTime=T_MAX,
        polarized=polarized,
        disableTransmission=disableTransmission,
    )
    # run pipeline
    pl.runPipeline(trace.collectStages())
    paths, length, codes = track.result(0)

    # check hits
    hits = rec.queue.view(0)
    # we placed both camera and light source in spheres
    # if there's no transmission allowed, there won't be any light paths
    if disableTransmission:
        assert hits.count == 0
        return  # nothing to test
    # check all other cases
    assert hits.count > 0
    assert hits.count <= trace.maxHits
    hits = hits[: hits.count]

    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX

    # sanity check polarization
    if polarized:
        # check stokes is normalized
        assert np.abs(hits["stokes"][:, 0] - 1.0).max() < 1e-6
        assert np.square(hits["stokes"][:, 1:]).sum(-1).max() <= 1.0
        assert hits["stokes"][:, 1:].max() <= 1.0
        assert hits["stokes"][:, 1:].min() >= -1.0
        # check polarization ref is perpendicular to plane of incidence and normalized
        # note: this only works because we use spheres and only scale and translate them
        polRef = hits["polarizationRef"]
        assert np.abs((polRef * hits["normal"]).sum(-1)).max() < 1e-6
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        # for pencil camera all polRefs are degenerate so nothing to test
        # polRef_exp = np.cross(hits["direction"], hits["normal"])
        # d = np.square(polRef_exp).sum(-1)
        # mask = d > 1e-7
        # polRef_exp /= np.sqrt(d)[:, None]
        # # for very small angle we reuse the old polRef to minimize error
        # assert np.abs(np.abs((polRef_exp * polRef).sum(-1))[mask] - 1.0).max() < 1e-6


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
    t1 = Transform.TRS(scale=r, translate=(17.0, 17.0, 17.0))
    c1 = store.createInstance("sphere", "mat", t1, detectorId=0)
    t2 = Transform.TRS(scale=r, translate=(-17.0, -17.0, -17.0))
    c2 = store.createInstance("sphere", "mat", t2, detectorId=1)
    t3 = Transform.TRS(scale=r, translate=(17.0, 17.0, -17.0))
    c3 = store.createInstance("sphere", "vol", t3)
    t4 = Transform.TRS(scale=r, translate=(-17.0, 17.0, 17.0))
    c4 = store.createInstance("sphere", "abs", t4)
    bbox = theia.scene.RectBBox((-50.0,) * 3, (50.0,) * 3)
    scene = theia.scene.Scene(
        [c1, c2, c3, c4], matStore.material, medium=matStore.media["water"], bbox=bbox
    )

    # create pipeline
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    photons = theia.light.UniformWavelengthSource()
    source = theia.light.SphericalLightSource(timeRange=(T0, T1))
    response = theia.response.EmptyResponse()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneForwardTracer(
        N,
        source,
        photons,
        response,
        rng,
        callback=stats,
        scene=scene,
        targetIdx=1,
        maxPathLength=1,
        scatterCoefficient=0.005 / u.m,
        maxTime=100.0 * u.ns,
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
    assert stats.maxIter == 0
    assert stats.error == 0
    assert stats.mismatch == 0
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check stats again
    assert stats.created == N
    assert stats.absorbed > 0
    assert stats.hit > 0
    assert stats.detected > 0
    assert stats.scattered > 0
    assert stats.lost > 0
    assert stats.decayed > 0
    assert stats.volume > 0
    assert stats.maxIter > 0
    assert stats.error == 0
    assert stats.mismatch == 0
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
    assert stats.maxIter == 0
    assert stats.error == 0
    assert stats.mismatch == 0


@pytest.mark.parametrize("polarized", [True, False])
@pytest.mark.parametrize("polarizedTrack", [True, False])
def test_TrackRecordCallback(polarizedTrack: bool, polarized: bool):
    N = 32 * 256
    N_SCATTER = 6
    LENGTH = N_SCATTER + 2  # one more than needed
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 500.0 * u.ns
    target_pos, target_radius = (5.0, 2.0, -8.0) * u.m, 4.0 * u.m
    light_pos = (100.0, -50.0, 20.0) * u.m
    light_dir = (1.0, 0.0, 0.0)
    stokes = (1.0, 0.5, 0.1, 0.4)
    polRef = (0.0, 1.0, 0.0)

    # create water medium
    water = WaterModel().createMedium()
    store = theia.material.MaterialStore([], media=[water])

    # create pipeline
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    photons = theia.light.UniformWavelengthSource()
    source = theia.light.PencilLightSource(
        position=light_pos,
        direction=light_dir,
        timeRange=(T0, T1),
        stokes=stokes,
        polarizationRef=polRef,
    )
    response = theia.response.EmptyResponse()
    track = theia.trace.TrackRecordCallback(N, LENGTH, polarized=polarizedTrack)
    tracer = theia.trace.VolumeForwardTracer(
        N,
        source,
        SphereTarget(position=target_pos, radius=target_radius),
        photons,
        response,
        rng,
        callback=track,
        medium=store.media["water"],
        nScattering=N_SCATTER,
        scatterCoefficient=0.1,
        maxTime=T_MAX,
        traceBBox=theia.scene.RectBBox((-200.0 * u.m,) * 3, (200.0 * u.m,) * 3),
        polarized=polarized,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check result
    tracks, lengths, codes = track.result(0)
    assert tracks.shape == (N, LENGTH, 11 if polarizedTrack else 4)
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

    # check polarization
    if polarizedTrack and polarized:
        # only check first one. For later steps we don't know the expected state
        assert np.isclose(tracks[:, 0, 4:8], stokes).all()
        assert np.isclose(tracks[:, 0, 8:], polRef).all()
    elif polarizedTrack:
        # all states should have been saved as unpolarized with zero reference
        assert np.isclose(tracks[:, 0, 4:], (1.0, *(0.0,) * 6)).all()


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
    trafo = Transform.TRS(scale=50.0, translate=(75.0, 0.0, 0.0))
    cube = store.createInstance("cube", "mat", trafo)
    scene = theia.scene.Scene([cube], matStore.material)

    # create scene
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    photons = theia.light.UniformWavelengthSource(
        lambdaRange=(LAMBDA, LAMBDA),  # const lambda
    )
    source = theia.light.PencilLightSource(
        position=(0.0, 0.0, 0.0),
        # important: hit cube not straight on but in angle to test for no refraction
        direction=(0.8, 0.36, 0.48),
        timeRange=(0.0, 0.0),  # const time
        budget=1.0,
    )
    estimator = theia.response.EmptyResponse()
    tracker = theia.trace.TrackRecordCallback(N, 4)
    tracer = theia.trace.SceneForwardTracer(
        N,
        source,
        photons,
        estimator,
        rng,
        callback=tracker,
        scene=scene,
        targetIdx=1,
        maxPathLength=2,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

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
    r"""
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
    splitter_trans = Transform.TRS(
        scale=(0.1, 5.0, 5.0),
        rotate=(0.0, 0.0, 1.0, 45.0),
        translate=(0.141, 0.0, 0.0),
    )
    splitter = store.createInstance("cube", "mat", splitter_trans)
    ref_trans = Transform.TRS(scale=(50.0, 0.5, 50.0), translate=(0.0, -50.0, 0.0))
    ref_det = store.createInstance("cube", "det", ref_trans, detectorId=1)
    trans_trans = Transform.TRS(scale=(0.5, 50.0, 50.0), translate=(50.0, 0.0, 0.0))
    trans_det = store.createInstance("cube", "det", trans_trans, detectorId=2)
    scene = theia.scene.Scene([splitter, trans_det, ref_det], matStore.material)

    # create pipeline
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    photons = theia.light.UniformWavelengthSource(
        lambdaRange=(500.0, 500.0) * u.nm,  # const lambda
    )
    source = theia.light.PencilLightSource(
        position=(-20.0, 0.0, 0.0) * u.m,
        direction=(1.0, 0.0, 0.0),
        timeRange=(0.0, 0.0),  # const time
        budget=1.0,
    )
    recorder = theia.response.HitRecorder()
    tracer = theia.trace.SceneForwardTracer(
        N,
        source,
        photons,
        recorder,
        rng,
        scene=scene,
        targetIdx=1,
        maxPathLength=4,
    )
    pipeline = pl.Pipeline(tracer.collectStages())
    # run pipeline
    pipeline.run(0)
    hits_ref = recorder.queue.view(0)
    # run for second detector
    tracer.setParam("targetIdx", 2)
    pipeline.run(1)
    hits_trans = recorder.queue.view(1)

    # check result
    assert hits_ref.count + hits_trans.count == N
    reflectance_sim = hits_ref.count / N
    assert np.abs(reflectance_sim - reflectance) <= err


@pytest.mark.parametrize("polarized", [True, False])
def test_DirectTracer_volume(polarized: bool):
    N = 32 * 256
    # params
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 45.0 * u.ns
    camDir = (1.0, 0.0, 0.0)
    camPos = (5.0, 2.0, -1.0)
    camUp = (0.0, 0.0, 1.0)
    width, length = 50.0 * u.cm, 40.0 * u.cm
    lightPos = (10.0, 5.0, 2.0)
    lightBudget = 1000.0

    # create water medium
    water = WaterModel().createMedium()
    store = theia.material.MaterialStore([], media=[water])

    # estimate min time
    v_max = np.max(water.group_velocity)
    d = np.array(lightPos) - np.array(camPos)
    d = np.multiply(d, camDir).sum(-1)  # not exact, but lower bound
    t_min = d / v_max + T0

    # create pipeline
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.UniformWavelengthSource()
    light = theia.light.SphericalLightSource(
        position=lightPos, budget=lightBudget, timeRange=(T0, T1)
    )
    camera = theia.camera.FlatCamera(
        width=width,
        length=length,
        position=camPos,
        direction=camDir,
        up=camUp,
    )
    recorder = theia.response.HitRecorder(polarized=polarized)
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.DirectLightTracer(
        N,
        light,
        camera,
        photons,
        recorder,
        rng,
        callback=stats,
        medium=store.media["water"],
        maxTime=T_MAX,
        polarized=polarized,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check hits
    hits = recorder.queue.view(0)
    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]
    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX

    # sanity check polarization
    if polarized:
        # check stokes is normalized
        assert np.abs(hits["stokes"][:, 0] - 1.0).max() < 1e-6
        assert np.square(hits["stokes"][:, 1:]).sum(-1).max() <= 1.0
        assert hits["stokes"][:, 1:].max() <= 1.0
        assert hits["stokes"][:, 1:].min() >= -1.0
        # check polarization ref is perpendicular to plane of incidence and normalized
        polRef = hits["polarizationRef"]
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        polRef_exp = np.cross(hits["direction"], hits["normal"])
        d = np.square(polRef_exp).sum(-1)
        mask = d > 1e-7
        polRef_exp /= np.sqrt(d)[:, None]
        # for very small angle we reuse the old polRef to minimize error
        assert np.abs(np.abs((polRef_exp * polRef).sum(-1))[mask] - 1.0).max() < 1e-6

    # check stats
    assert stats.created == tracer.batchSize
    assert stats.detected + stats.decayed + stats.absorbed == tracer.batchSize
    assert stats.detected == len(hits)


@pytest.mark.parametrize("polarized", [True, False])
def test_DirectTracer_scene(polarized: bool):
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 256
    # same as volume version, but now put something before camera to test shadow rays
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 45.0 * u.ns
    camDir = (1.0, 0.0, 0.0)
    camPos = (5.0, 2.0, -1.0)
    camUp = (0.0, 0.0, 1.0)
    width, length = 50.0 * u.cm, 40.0 * u.cm
    lightPos = (10.0, 5.0, 2.0)
    lightBudget = 1000.0

    # create water medium
    water = WaterModel().createMedium()
    absorber = theia.material.Material("absorber", None, water, flags="B")
    store = theia.material.MaterialStore([absorber], media=[water])
    # create scene
    # block half of the camera to see whether shadow rays work properly
    meshes = theia.scene.MeshStore({"cube": "assets/cube.ply"})
    aperture_size = (1.0 * u.cm, width, length)
    aperture_pos = (5.03, 2.0 + width, -1.0)
    t = Transform.TRS(scale=aperture_size, translate=aperture_pos)
    a = meshes.createInstance("cube", "absorber", t)
    scene = theia.scene.Scene([a], store.material, medium=store.media["water"])

    # estimate min time
    v_max = np.max(water.group_velocity)
    d = np.array(lightPos) - np.array(camPos)
    d = np.multiply(d, camDir).sum(-1)  # not exact, but lower bound
    t_min = d / v_max + T0

    # create pipeline stages
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.UniformWavelengthSource()
    light = theia.light.SphericalLightSource(
        position=lightPos, budget=lightBudget, timeRange=(T0, T1)
    )
    camera = theia.camera.FlatCamera(
        width=width,
        length=length,
        position=camPos,
        direction=camDir,
        up=camUp,
    )
    recorder = theia.response.HitRecorder(polarized=polarized)
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.DirectLightTracer(
        N,
        light,
        camera,
        photons,
        recorder,
        rng,
        scene,
        callback=stats,
        maxTime=T_MAX,
        polarized=polarized,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check hits
    hits = recorder.queue.view(0)
    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]
    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX
    # check we hit only half of the detector
    assert np.abs(hits["position"].max(0) - (0.0, length / 2, 0.0)).max() < 0.025
    assert np.abs(hits["position"].min(0) - (-width / 2, -length / 2, 0.0)).max() < 5e-4

    # sanity check polarization
    if polarized:
        # check stokes is normalized
        assert np.abs(hits["stokes"][:, 0] - 1.0).max() < 1e-6
        assert np.square(hits["stokes"][:, 1:]).sum(-1).max() <= 1.0
        assert hits["stokes"][:, 1:].max() <= 1.0
        assert hits["stokes"][:, 1:].min() >= -1.0
        # check polarization ref is perpendicular to plane of incidence and normalized
        polRef = hits["polarizationRef"]
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        polRef_exp = np.cross(hits["direction"], hits["normal"])
        d = np.square(polRef_exp).sum(-1)
        mask = d > 1e-7
        polRef_exp /= np.sqrt(d)[:, None]
        # for very small angle we reuse the old polRef to minimize error
        assert np.abs(np.abs((polRef_exp * polRef).sum(-1))[mask] - 1.0).max() < 1e-6

    # check stats
    assert stats.created == tracer.batchSize
    assert stats.detected + stats.decayed + stats.absorbed == tracer.batchSize
    assert stats.detected == len(hits)
    assert stats.absorbed > 0
