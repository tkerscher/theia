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

from common.models import WaterModel


def test_VolumeTracer():
    N = 32 * 256
    N_LAMBDA = 4
    N_SCATTER = 6
    T0, T1 = 10.0, 20.0
    T_MAX = 500.0
    light_pos = (-1.0, -7.0, 0.0)
    light_intensity = 1000.0
    target_pos, target_radius = (5.0, 2.0, -8.0), 4.0

    # create water medium
    water = WaterModel().createMedium()
    tensor, _, media = theia.material.bakeMaterials(media=[water])

    # estimate max speed
    d_min = (
        np.sqrt(np.square(np.subtract(light_pos, target_pos)).sum(-1)) - target_radius
    )
    v_max = np.max(water.group_velocity)
    t_min = d_min / v_max + T0  # ns

    # create pipeline
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    source = theia.light.SphericalLightSource(
        nLambda=N_LAMBDA,
        position=light_pos,
        intensity=light_intensity,
        timeRange=(T0, T1),
    )
    recorder = theia.estimator.HitRecorder(N * N_SCATTER * N_LAMBDA)
    tracer = theia.trace.VolumeTracer(
        N,
        source,
        recorder,
        rng,
        medium=media["water"],
        nScattering=N_SCATTER,
        maxTime=T_MAX,
        target=theia.scene.SphereBBox(target_pos, target_radius),
    )
    # run pipeline
    pl.runPipeline([rng, source, tracer, recorder])

    # check hits
    hits = recorder.view(0)
    assert hits.count > 0
    hits = hits[: hits.count]

    assert np.allclose(np.square(hits["position"]).sum(-1), target_radius**2)
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX
    # TODO: more sophisticated tests...


@pytest.mark.parametrize("vol", [True, False])
@pytest.mark.parametrize("trans", [True, False])
def test_SceneShadowTracer(vol: bool, trans: bool):
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 256
    N_LAMBDA = 4
    N_SCATTER = 6
    T0, T1 = 10.0, 20.0
    T_MAX = 500.0
    light_pos = (-1.0, -7.0, 0.0)
    light_intensity = 1000.0

    # create materials
    water = WaterModel().createMedium()
    glass = theia.material.BK7Model().createMedium()
    mat = theia.material.Material("mat", glass, water, flags=("D", "B"))
    tensor, material, media = theia.material.bakeMaterials(mat)
    # create scene
    store = theia.scene.MeshStore(
        {"cube": "assets/cone.stl", "sphere": "assets/sphere.stl"}
    )
    r, d = 40.0, 5.0
    r_scale = 0.99547149974733  # radius of inscribed sphere (icosphere)
    r_insc = r * r_scale
    x, y, z = 10.0, 5.0, -5.0
    t1 = theia.scene.Transform.Scale(r, r, r).translate(x, y, z + r + d)
    c1 = store.createInstance("sphere", "mat", transform=t1, detectorId=0)
    t2 = theia.scene.Transform.Scale(r, r, r).translate(x, y, z - r - d)
    c2 = store.createInstance("sphere", "mat", transform=t2, detectorId=1)
    detectors = [
        theia.scene.SphereBBox((x, y, z + r + d), r),
        theia.scene.SphereBBox((x, y, z - r - d), r),
    ]
    scene = theia.scene.Scene(
        [c1, c2], material, medium=media["water"], detectors=detectors
    )

    # calculate min time
    target_pos = (x, y, z - r - d)  # detector #1
    d_min = np.sqrt(np.square(np.subtract(target_pos, light_pos)).sum(-1)) - r
    v_max = np.max(water.group_velocity)
    t_min = d_min / v_max + T0  # ns

    # create pipeline stages
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    source = theia.light.SphericalLightSource(
        nLambda=N_LAMBDA,
        position=light_pos,
        intensity=light_intensity,
        timeRange=(T0, T1),
    )
    recorder = theia.estimator.HitRecorder(N * N_SCATTER * N_LAMBDA)
    tracer = theia.trace.SceneShadowTracer(
        N,
        source,
        recorder,
        rng,
        scene=scene,
        nScattering=N_SCATTER,
        targetIdx=1,
        maxTime=T_MAX,
        disableVolumeBorder=vol,
        disableTransmission=trans,
    )
    # run pipeline
    pl.runPipeline([rng, source, tracer, recorder])

    # check hits
    hits = recorder.view(0)
    assert hits.count > 0
    hits = hits[: hits.count]

    r_hits = np.sqrt(np.square(hits["position"]).sum(-1))
    assert np.all((r_hits <= 1.0) & (r_hits >= r_scale))
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX


@pytest.mark.parametrize("vol", [True, False])
@pytest.mark.parametrize("trans", [True, False])
def test_SceneWalkTracer(vol: bool, trans: bool):
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 256
    N_LAMBDA = 4
    N_SCATTER = 6
    T0, T1 = 10.0, 20.0
    T_MAX = 500.0
    light_pos = (-1.0, -7.0, 0.0)
    light_intensity = 1000.0

    # create materials
    water = WaterModel().createMedium()
    glass = theia.material.BK7Model().createMedium()
    mat = theia.material.Material("mat", glass, water, flags=("D", "B"))
    tensor, material, media = theia.material.bakeMaterials(mat)
    # create scene
    store = theia.scene.MeshStore(
        {"cube": "assets/cone.stl", "sphere": "assets/sphere.stl"}
    )
    r, d = 40.0, 5.0
    r_scale = 0.99547149974733  # radius of inscribed sphere (icosphere)
    r_insc = r * r_scale
    x, y, z = 10.0, 5.0, -5.0
    t1 = theia.scene.Transform.Scale(r, r, r).translate(x, y, z + r + d)
    c1 = store.createInstance("sphere", "mat", transform=t1, detectorId=0)
    t2 = theia.scene.Transform.Scale(r, r, r).translate(x, y, z - r - d)
    c2 = store.createInstance("sphere", "mat", transform=t2, detectorId=1)
    detectors = [
        theia.scene.SphereBBox((x, y, z + r + d), r),
        theia.scene.SphereBBox((x, y, z - r - d), r),
    ]
    scene = theia.scene.Scene(
        [c1, c2], material, medium=media["water"], detectors=detectors
    )

    # calculate min time
    target_pos = (x, y, z - r - d)  # detector #1
    d_min = np.sqrt(np.square(np.subtract(target_pos, light_pos)).sum(-1)) - r
    v_max = np.max(water.group_velocity)
    t_min = d_min / v_max + T0  # ns

    # create pipeline stages
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    source = theia.light.SphericalLightSource(
        nLambda=N_LAMBDA,
        position=light_pos,
        intensity=light_intensity,
        timeRange=(T0, T1),
    )
    recorder = theia.estimator.HitRecorder(N * N_SCATTER * N_LAMBDA)
    tracer = theia.trace.SceneWalkTracer(
        N,
        source,
        recorder,
        rng,
        scene=scene,
        nScattering=N_SCATTER,
        targetSampleProb=0.4,
        targetIdx=1,
        maxTime=T_MAX,
        disableVolumeBorder=vol,
        disableTransmission=trans,
    )
    # run pipeline
    pl.runPipeline([rng, source, tracer, recorder])

    # check hits
    hits = recorder.view(0)
    assert hits.count > 0
    hits = hits[: hits.count]

    r_hits = np.sqrt(np.square(hits["position"]).sum(-1))
    assert np.all((r_hits <= 1.0) & (r_hits >= r_scale))
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX
