import pytest

import hephaistos as hp
import hephaistos.pipeline as pl
import numpy as np

from itertools import product

from theia.camera import FlatCamera, PointCamera, SphereCamera
from theia.device import isRayTracingEnabled
from theia.light import PencilLightSource, SphericalLightSource, UniformWavelengthSource
from theia.material import Material, MaterialStore, getPropertySamples, VACUUM_IDX
from theia.model import BK7Model, PureWaterModel
from theia.random import PhiloxRNG
from theia.ray import UnpolarizedRay
from theia.response import HitRecorder
from theia.scene import MeshStore, Scene, Transform
from theia.surface import AbsorbingSurface, BorderSurface, DielectricSurface
from theia.target import SphereTarget, SphereTargetGuide
from theia.volume import Attenuating, Transparent
import theia.trace
import theia.units as u


def test_TracerReportsConfig():
    # Simple check whether tracer calls `prepare()` and `updateConfig` on the
    # given response

    batchSize = 64 * 1024
    batchSize2 = 96 * 1024
    capacity = 128 * 1024

    class MockResponse(theia.response.EmptyResponse):

        def __init__(self) -> None:
            super().__init__()
            self.prepareConfig = None
            self.updatedConfig = None
            self.ray = None

        def prepare(self, config, ray):
            self.prepareConfig = config
            self.ray = ray

        def updateConfig(self, config):
            self.updatedConfig = config

    vol = Attenuating()
    medium = theia.model.PureWaterModel().createMedium(physicModel=vol)
    store = theia.material.MaterialStore([], media=[medium])

    ray = UnpolarizedRay()
    philox = theia.random.PhiloxRNG(key=0xABBA)
    target = theia.target.SphereTarget()
    photons = theia.light.ConstWavelengthSource()
    source = theia.light.SphericalLightSource(photons, mediumIdx=VACUUM_IDX)
    response = MockResponse()
    tracer = theia.trace.VolumeForwardTracer(
        batchSize,
        ray,
        source,
        target,
        response,
        rng=philox,
        materials=store,
        volume=vol,
        capacity=capacity,
    )

    # check prepare was called with the correct config
    assert response.ray == ray
    assert response.prepareConfig is not None
    assert response.prepareConfig.batchSize == batchSize
    assert response.prepareConfig.capacity == capacity
    assert response.prepareConfig.maxHitsPerRay == tracer.maxHitsPerRay
    assert response.prepareConfig.normalization == tracer.normalization
    assert response.prepareConfig.normalization == 1.0 / batchSize

    # update batch size
    tracer.batchSize = batchSize2
    tracer.update(0)
    # check updateConfig was called with the correct config
    assert response.updatedConfig is not None
    assert response.updatedConfig.batchSize == batchSize2
    assert response.updatedConfig.capacity == capacity
    assert response.updatedConfig.maxHitsPerRay == tracer.maxHitsPerRay
    assert response.updatedConfig.normalization == tracer.normalization
    assert response.updatedConfig.normalization == 1.0 / batchSize2


def test_TrackRecordCallback():
    N = 32 * 256
    MAX_LENGTH = 8
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 500.0 * u.ns
    target_pos, target_radius = (5.0, 2.0, -8.0) * u.m, 4.0 * u.m
    light_pos = (100.0, -50.0, 20.0) * u.m
    light_dir = (1.0, 0.0, 0.0)

    # create water medium
    water = PureWaterModel().createMedium(physicModel=Attenuating())
    materials = MaterialStore([], media=[water])

    # create pipeline
    rng = PhiloxRNG(key=0xABBA)
    ray = UnpolarizedRay()
    photons = UniformWavelengthSource()
    source = PencilLightSource(
        photons,
        mediumIdx=materials.media["water"],
        position=light_pos,
        direction=light_dir,
        timeRange=(T0, T1),
    )
    response = theia.response.EmptyResponse()
    track = theia.trace.TrackRecordCallback(N, MAX_LENGTH)
    tracer = theia.trace.VolumeForwardTracer(
        N,
        ray,
        source,
        SphereTarget(position=target_pos, radius=target_radius),
        response,
        rng,
        materials,
        volume=materials.media["water"],
        callback=track,
        maxPathLength=MAX_LENGTH,
        sampleCoefficient=0.1,
        maxTime=T_MAX,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check results
    tracks, lengths, codes = track.result(0)
    assert tracks.shape == (N, MAX_LENGTH, 4)
    assert np.all(tracks[:, 0, :3] == light_pos)
    assert tracks[:, 0, 3].min() >= T0 and lengths.max() <= MAX_LENGTH
    # check result codes in range
    minCode = min(code.value for code in theia.trace.EventResultCode)
    maxCode = max(code.value for code in theia.trace.EventResultCode)
    # codes after the individual path length are undefined and have to be filtered out
    for i in range(N):
        c = codes[i, : lengths[i].item()]
        assert minCode <= c.min() and c.max() <= maxCode


def _parameterize_VolumeForwardTracer():
    configs = product([True, False], repeat=6)
    for direct, target, scatter, limitTime, particle, transient in configs:
        # skip unnecessary configs
        if limitTime and not transient:
            continue
        if particle and target:
            continue
        if particle and not scatter:
            continue
        # return accepted config
        yield (not direct, not target, not scatter, limitTime, particle, transient)


def _test_VolumeForwardTracer_id():
    configs = _parameterize_VolumeForwardTracer()
    for disDirect, disTarget, disScatter, limitTime, particle, transient in configs:
        args = []
        if not disDirect:
            args.append("direct")
        if not disTarget:
            args.append("target")
        if not disScatter:
            args.append("scatter")
        if limitTime:
            args.append("limitTime")
        args.append("particle" if particle else "ray")
        args.append("transient" if transient else "instant")
        yield "-".join(args)


@pytest.mark.parametrize(
    "disableDirect,disableTarget,disableScattering,limitTime,particle,transient",
    list(_parameterize_VolumeForwardTracer()),
    ids=list(_test_VolumeForwardTracer_id()),
)
def test_VolumeForwardTracer(
    disableDirect: bool,
    disableTarget: bool,
    disableScattering: bool,
    limitTime: bool,
    particle: bool,
    transient: bool,
):
    N = 32 * 256
    N_SCATTER = 6
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 1.0 * u.us if limitTime else 100.0 * u.us
    capacity = 48 * 256
    objectId = 42
    light_pos = (-1.0, -7.0, 0.0) * u.m
    light_budget = 1000.0
    target_pos, target_radius = (5.0, 2.0, -8.0) * u.m, 4.0 * u.m

    # create water medium
    water = PureWaterModel().createMedium(physicModel=Attenuating())
    materials = MaterialStore([], media=[water])

    # estimate max speed
    d_min = (
        np.sqrt(np.square(np.subtract(light_pos, target_pos)).sum(-1)) - target_radius
    )
    vg = getPropertySamples(water, "group_velocity")
    assert vg is not None
    v_max = np.max(vg)
    t_min = d_min / v_max + T0

    # create pipeline
    rng = PhiloxRNG(key=0xC01DC0FFEE)
    ray = UnpolarizedRay(transient=transient, particle=particle)
    photons = UniformWavelengthSource()
    source = SphericalLightSource(
        photons,
        mediumIdx=materials.media["water"],
        position=light_pos,
        timeRange=(T0, T1),
        budget=light_budget,
        emitParticles=particle,
    )
    target = SphereTarget(position=target_pos, radius=target_radius)
    recorder = HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.VolumeForwardTracer(
        N,
        ray,
        source,
        target,
        recorder,
        rng,
        materials,
        volume="water",
        capacity=capacity,
        callback=stats,
        maxPathLength=N_SCATTER,
        objectId=objectId,
        sampleCoefficient=0.0 if disableScattering else float("NaN"),
        maxTime=T_MAX,
        disableDirectLighting=disableDirect,
        disableTargetSampling=disableTarget,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check hits
    assert recorder.queue is not None
    hits = recorder.queue.view(0)
    if disableDirect and disableScattering:
        # this combination makes it impossible for the tracer to create hits
        assert hits.count == 0
        assert stats.scattered == 0
        return  # nothing more to check
    else:
        assert hits.count > 0
    assert hits.count <= tracer.maxHits

    assert np.allclose(np.square(hits["position"]).sum(-1), 1.0)
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.all(hits["objectId"] == objectId)
    if transient:
        assert np.min(hits["time"]) >= t_min
        assert np.max(hits["time"]) <= T_MAX

    # check config via stats
    assert stats.created == tracer.batchSize
    if disableScattering:
        assert stats.scattered == 0
    else:
        assert stats.scattered > 0
    if disableDirect and disableTarget:
        assert stats.absorbed > 0
        assert stats.detected > 0
    elif disableDirect and not disableTarget:
        assert stats.absorbed > 0
        assert stats.detected == 0
    elif not disableDirect and disableTarget:
        assert stats.detected > 0
        if particle:
            assert stats.absorbed > 0
        else:
            assert stats.absorbed == 0
    elif not disableDirect and not disableTarget:
        assert stats.absorbed > 0
        assert stats.detected == 0

    # TODO: more sophisticated tests...


@pytest.mark.parametrize("disableDirect", [True, False])
@pytest.mark.parametrize("disableTarget", [True, False])
@pytest.mark.parametrize("disableScattering", [True, False])
@pytest.mark.parametrize("limitTime", [True, False])
def test_VolumeBackwardTracer(
    disableDirect: bool, disableTarget: bool, disableScattering: bool, limitTime: bool
) -> None:
    N = 32 * 256
    N_SCATTER = 6
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 1.0 * u.us
    capacity = 48 * 256
    light_pos = (-1.0, -7.0, 0.0) * u.m
    light_budget = 1000.0
    target_pos, target_radius = (5.0, 2.0, -8.0) * u.m, 4.0 * u.m
    target = SphereTarget(position=target_pos, radius=target_radius)

    # create water medium
    water = PureWaterModel().createMedium(physicModel=Attenuating())
    materials = MaterialStore([], media=[water])
    medIdx = materials.media["water"]

    # estimate max speed
    d_min = (
        np.sqrt(np.square(np.subtract(light_pos, target_pos)).sum(-1)) - target_radius
    )
    vg = theia.material.getPropertySamples(water, "group_velocity")
    assert vg is not None
    v_max = np.max(vg)
    t_min = d_min / v_max + T0

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC01DC0FFEE)
    photons = UniformWavelengthSource()
    light = SphericalLightSource(
        mediumIdx=medIdx,
        position=light_pos,
        timeRange=(T0, T1),
        budget=light_budget,
    )
    camera = SphereCamera(mediumIdx=medIdx, position=target_pos, radius=target_radius)
    recorder = HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.VolumeBackwardTracer(
        N,
        ray,
        light,
        camera,
        photons,
        recorder,
        philox,
        materials,
        volume="water",
        capacity=capacity,
        callback=stats,
        maxPathLength=N_SCATTER,
        sampleCoefficient=0.0 if disableScattering else float("NaN"),
        target=None if disableTarget else target,
        maxTime=T_MAX if limitTime else float("inf"),
        disableDirectLighting=disableDirect,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check hits
    assert recorder.queue is not None
    hits = recorder.queue.view(0)
    if disableDirect and disableScattering:
        # this combination makes it impossible for the tracer to create hits
        assert hits.count == 0
        assert stats.scattered == 0
        return  # nothing more to check
    else:
        assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]

    assert np.allclose(np.square(hits["position"]).sum(-1), 1.0)
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.min(hits["time"]) >= t_min
    if limitTime:
        assert np.max(hits["time"]) <= T_MAX
    elif not disableScattering:
        # without scattering rays will immediately leave the volume
        assert np.max(hits["time"]) > T_MAX

    # check config via stats
    if disableScattering:
        assert stats.scattered == 0
    else:
        assert stats.scattered > 0
    if disableTarget or disableScattering:
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


@pytest.mark.parametrize("limitTime", [True, False])
def test_VolumeDirectTracer(limitTime: bool):
    N = 32 * 256
    # params
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 45.0 * u.ns
    capacity = 48 * 256
    camDir = (1.0, 0.0, 0.0)
    camPos = (5.0, 2.0, -1.0)
    camUp = (0.0, 0.0, 1.0)
    width, length = 50.0 * u.cm, 40.0 * u.cm
    lightPos = (10.0, 5.0, 2.0)
    lightBudget = 1000.0

    # create water medium
    water = PureWaterModel().createMedium(physicModel=Attenuating())
    materials = MaterialStore([], media=[water])
    waterIdx = materials.media["water"]

    # estimate min time
    vg = getPropertySamples(water, "group_velocity")
    assert vg is not None
    v_max = np.max(vg)
    d = np.array(lightPos) - np.array(camPos)
    d = np.multiply(d, camDir).sum(-1)  # not exact, but lower bound
    t_min = d / v_max + T0

    # create pipeline
    rng = PhiloxRNG(key=0xC0FFEE)
    ray = UnpolarizedRay()
    photons = UniformWavelengthSource()
    light = SphericalLightSource(
        position=lightPos,
        budget=lightBudget,
        timeRange=(T0, T1),
        mediumIdx=waterIdx,
    )
    camera = FlatCamera(
        mediumIdx=waterIdx,
        width=width,
        length=length,
        position=camPos,
        direction=camDir,
        up=camUp,
        objectId=10,
    )
    recorder = HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.VolumeDirectTracer(
        N,
        ray,
        light,
        camera,
        photons,
        recorder,
        rng,
        materials,
        volume="water",
        capacity=capacity,
        callback=stats,
        maxTime=T_MAX if limitTime else float("NaN"),
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check hits
    assert recorder.queue is not None
    hits = recorder.queue.view(0)
    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]
    assert np.all(hits["objectId"] == 10)
    assert np.min(hits["time"]) >= t_min
    if limitTime:
        assert np.max(hits["time"]) <= T_MAX
    else:
        assert np.max(hits["time"]) > T_MAX

    # check stats
    assert stats.created == tracer.batchSize
    assert stats.detected + stats.decayed + stats.absorbed == tracer.batchSize
    assert stats.detected == len(hits)


@pytest.mark.parametrize("limitTime", [True, False])
def test_SceneDirectTracer(limitTime: bool):
    if not isRayTracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 256
    # same as volume version, but now put something before camera to test shadow rays
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 45.0 * u.ns
    capacity = 48 * 256
    camDir = (1.0, 0.0, 0.0)
    camPos = (5.0, 2.0, -1.0)
    camUp = (0.0, 0.0, 1.0)
    width, length = 50.0 * u.cm, 40.0 * u.cm
    lightPos = (10.0, 5.0, 2.0)
    lightBudget = 1000.0

    # create water medium
    attenuate = Attenuating()
    water = PureWaterModel().createMedium(physicModel=attenuate)
    absorber = AbsorbingSurface()
    mat = Material("absorber", None, water, absorber)
    matStore = MaterialStore([mat], media=[water])
    waterIdx = matStore.media["water"]
    # create scene
    # block half of the camera to see whether shadow rays work properly
    meshes = theia.scene.MeshStore({"cube": "assets/cube.ply"})
    aperture_size = (1.0 * u.cm, width, length)
    aperture_pos = (5.03, 2.0 + width, -1.0)
    t = Transform.TRS(scale=aperture_size, translate=aperture_pos)
    a = meshes.createInstance("cube", "absorber", t)
    scene = Scene([a], matStore)

    # estimate min time
    vg = getPropertySamples(water, "group_velocity")
    assert vg is not None
    v_max = np.max(vg)
    d = np.array(lightPos) - np.array(camPos)
    d = np.multiply(d, camDir).sum(-1)  # not exact, but lower bound
    t_min = d / v_max + T0

    # create pipeline
    rng = PhiloxRNG(key=0xC0FFEE)
    ray = UnpolarizedRay()
    photons = UniformWavelengthSource()
    light = SphericalLightSource(
        position=lightPos,
        budget=lightBudget,
        timeRange=(T0, T1),
        mediumIdx=waterIdx,
    )
    camera = FlatCamera(
        mediumIdx=waterIdx,
        width=width,
        length=length,
        position=camPos,
        direction=camDir,
        up=camUp,
        objectId=10,
    )
    recorder = HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneDirectTracer(
        N,
        ray,
        light,
        camera,
        photons,
        recorder,
        rng,
        scene,
        capacity=capacity,
        callback=stats,
        maxTime=T_MAX if limitTime else float("inf"),
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check hits
    assert recorder.queue is not None
    hits = recorder.queue.view(0)
    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]
    assert np.all(hits["objectId"] == 10)
    assert np.min(hits["time"]) >= t_min
    if limitTime:
        assert np.max(hits["time"]) <= T_MAX
    else:
        assert np.max(hits["time"]) > T_MAX
    # check we hit only half of the detector
    assert np.abs(hits["position"].max(0) - (0.0, length / 2, 0.0)).max() < 0.025
    assert np.abs(hits["position"].min(0) - (-width / 2, -length / 2, 0.0)).max() < 5e-4
    # check stats
    assert stats.created == tracer.batchSize
    n = stats.detected + stats.decayed + stats.absorbed + stats.missed
    assert n == tracer.batchSize
    assert stats.detected == len(hits)
    assert stats.absorbed > 0


@pytest.mark.parametrize("inlineVolumeModels", [True, False])
@pytest.mark.parametrize("disableDirect", [True, False])
@pytest.mark.parametrize("disableScattering", [True, False])
def test_SceneBackwardTracer(
    inlineVolumeModels: bool,
    disableDirect: bool,
    disableScattering: bool,
) -> None:
    if not isRayTracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 256
    MAX_LENGTH = 8
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 100000.0 * u.us
    capacity = 48 * 256
    light_pos = (-1.0, -7.0, 0.0) * u.m
    light_budget = 1000.0

    # create materials
    attenuate = Attenuating()
    absorbing = Attenuating(absorb=True)
    water = PureWaterModel().createMedium(physicModel=attenuate)
    water2 = PureWaterModel().createMedium(physicModel=absorbing, name="water2")
    glass = BK7Model().createMedium()
    absorber = AbsorbingSurface()
    border = BorderSurface()
    dielectric = DielectricSurface()
    mat = Material("mat", glass, water, dielectric, flags=("DR", "B"))
    matAbs = Material("abs", None, water, absorber)
    matVol = Material("vol", water if inlineVolumeModels else water2, water, border)
    matStore = MaterialStore([mat, matAbs, matVol])
    waterIdx = matStore.media["water"]
    if inlineVolumeModels:
        assert len(matStore.volumeModels) == 2
    else:
        assert len(matStore.volumeModels) == 3
    # create scene
    # it will consist of 6 spheres arranged in an octahedron
    store = MeshStore({"cube": "assets/cube.ply", "sphere": "assets/sphere.stl"})
    r, d = 40.0 * u.m, 5.0 * u.m
    a = 60.0 * u.m  # >= (2r+ d) / sqrt(2)
    r_scale = 0.99547149974733 * u.m  # radius of inscribed sphere (icosphere)
    r_insc = r * r_scale
    x, y, z = 10.0, 5.0, -5.0
    _t = lambda dx, dy, dz: Transform.TRS(scale=r, translate=(x + dx, y + dy, z + dz))
    instances = [
        store.createInstance("sphere", "mat", _t(0, 0, a), detectorId=1),
        store.createInstance("sphere", "mat", _t(0, 0, -a), detectorId=2),
        store.createInstance("sphere", "abs", _t(0, a, 0)),
        store.createInstance("sphere", "abs", _t(0, -a, 0)),
        store.createInstance("sphere", "vol", _t(a, 0, 0)),
        store.createInstance("sphere", "vol", _t(-a, 0, 0)),
    ]
    scene = Scene(instances, matStore)

    # calculate min time
    target_pos = (x, y, z + a) * u.m  # detector #1
    d_min = np.sqrt(np.square(np.subtract(target_pos, light_pos)).sum(-1)) - r
    vg = theia.material.getPropertySamples(water, "group_velocity")
    assert vg is not None
    v_max = np.max(vg)
    t_min = d_min / v_max + T0

    # create pipeline
    rng = PhiloxRNG(key=0xC01DC0FFEE)
    ray = UnpolarizedRay()
    photons = UniformWavelengthSource()
    source = SphericalLightSource(
        mediumIdx=waterIdx,
        position=light_pos,
        timeRange=(T0, T1),
        budget=light_budget,
    )
    camera = SphereCamera(
        mediumIdx=waterIdx,
        position=target_pos,
        radius=r,
    )
    recorder = HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneBackwardTracer(
        N,
        ray,
        source,
        camera,
        photons,
        recorder,
        rng,
        scene,
        capacity=capacity,
        callback=stats,
        maxPathLength=MAX_LENGTH,
        sampleCoefficient=0.0 if disableScattering else float("NaN"),
        maxTime=T_MAX,
        disableDirectLighting=disableDirect,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())

    # check hits
    assert recorder.queue is not None
    hits = recorder.queue.view(0)

    if disableScattering and disableDirect:
        # We can only connect to the light source at scatter points -> expect no hits
        assert stats.scattered == 0
        assert hits.count == 0
        return  # nothing more to test
    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]

    assert np.max(hits["time"]) <= T_MAX
    assert np.min(hits["time"]) >= t_min

    # check config via stats
    assert stats.mismatch == 0
    assert stats.absorbed > 0
    assert stats.volume > 0
    if disableScattering:
        assert stats.scattered == 0
    else:
        assert stats.scattered > 0
    if disableDirect:
        assert stats.created == tracer.batchSize
        # there will only be hits from shadow rays
        assert stats.detected == 0
    else:
        assert stats.created == 2 * tracer.batchSize
        assert stats.detected > 0


@pytest.mark.parametrize(
    "particle,guide,targetId,inlineVolumeModels,disableDirect,disableScattering",
    [
        (True, False, False, True, False, False),
        (True, False, False, False, False, False),
        (True, False, True, True, False, False),
        (False, True, True, True, False, False),
        (False, True, True, False, False, True),
        (False, False, False, True, True, True),
        (False, True, True, False, False, False),
        (False, False, True, True, False, False),
        (False, False, True, False, False, False),
        (False, False, False, False, False, False),
        (False, False, True, False, True, False),
        (False, False, False, True, True, False),
        (False, False, True, False, False, True),
        (False, False, False, True, False, False),
    ],
)
def test_SceneForwardTracer(
    particle: bool,
    guide: bool,
    targetId: bool,
    inlineVolumeModels: bool,
    disableDirect: bool,
    disableScattering: bool,
):
    if not isRayTracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 256
    MAX_LENGTH = 10
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 1.0 * u.us
    capacity = 48 * 256
    light_pos = (-1.0, -7.0, -10.0) * u.m
    light_budget = 1000.0

    # create materials
    attenuate = Attenuating()
    absorbing = Attenuating(absorb=True)
    water = PureWaterModel().createMedium(physicModel=attenuate)
    water2 = PureWaterModel().createMedium(physicModel=absorbing, name="water2")
    glass = BK7Model().createMedium()
    absorber = AbsorbingSurface()
    border = BorderSurface()
    dielectric = DielectricSurface()
    mat = Material("mat", glass, water, dielectric, flags=("DR", "B"))
    matAbs = Material("abs", None, water, absorber)
    matVol = Material("vol", water if inlineVolumeModels else water2, water, border)
    matStore = MaterialStore([mat, matAbs, matVol])
    waterIdx = matStore.media["water"]
    if inlineVolumeModels:
        assert len(matStore.volumeModels) == 2
    else:
        assert len(matStore.volumeModels) == 3
    # create scene
    # it will consist of 6 spheres arranged in an octahedron
    store = MeshStore({"cube": "assets/cube.ply", "sphere": "assets/sphere.stl"})
    r, d = 40.0 * u.m, 5.0 * u.m
    a = 60.0 * u.m  # >= (2r+d) / sqrt(2) = min dist between spheres
    r_scale = 0.99547149974733 * u.m  # radius of inscribed sphere (icosphere)
    r_insc = r * r_scale
    x, y, z = 10.0, 5.0, -5.0
    _t = lambda dx, dy, dz: Transform.TRS(scale=r, translate=(x + dx, y + dy, z + dz))
    instances = [
        store.createInstance("sphere", "mat", _t(0, 0, a), detectorId=1),
        store.createInstance("sphere", "mat", _t(0, 0, -a), detectorId=2),
        store.createInstance("sphere", "abs", _t(0, a, 0)),
        store.createInstance("sphere", "abs", _t(0, -a, 0)),
        store.createInstance("sphere", "vol", _t(a, 0, 0)),
        store.createInstance("sphere", "vol", _t(-a, 0, 0)),
    ]
    scene = Scene(instances, matStore)

    targetGuide = SphereTargetGuide(position=(-x, -y, z + r + d), radius=r)

    # calculate min time
    if targetId:
        # detector #1 is farther away
        target_pos = (x, y, z + a) * u.m  # detector #1
        d_min = np.sqrt(np.square(np.subtract(target_pos, light_pos)).sum(-1)) - r
    else:
        target_pos = (x, y, z - a) * u.m  # detector #2
        d_min = np.sqrt(np.square(np.subtract(target_pos, light_pos)).sum(-1)) - r
    vg = theia.material.getPropertySamples(water, "group_velocity")
    assert vg is not None
    v_max = np.max(vg)
    t_min = d_min / v_max + T0

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xC01DC0FFEE)
    photons = UniformWavelengthSource()
    source = SphericalLightSource(
        photons,
        mediumIdx=waterIdx,
        position=light_pos,
        timeRange=(T0, T1),
        budget=light_budget,
        emitParticles=particle,
    )
    recorder = HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneForwardTracer(
        N,
        ray,
        source,
        recorder,
        rng,
        scene,
        capacity=capacity,
        callback=stats,
        maxPathLength=MAX_LENGTH,
        sampleCoefficient=0.0 if disableScattering else float("NaN"),
        maxTime=T_MAX,
        targetId=1 if targetId else None,
        targetGuide=targetGuide if guide else None,
        disableDirectLighting=disableDirect,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())
    hp.checkCurrentDeviceHealth()

    # check hits
    assert recorder.queue is not None
    hits = recorder.queue.view(0)

    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]

    # check config via stats
    assert stats.created == N
    assert stats.mismatch == 0
    assert stats.absorbed > 0
    assert stats.volume > 0
    if disableScattering:
        assert stats.scattered == 0
    else:
        assert stats.scattered > 0

    r_hits = np.sqrt(np.square(hits["position"]).sum(-1))
    assert np.all((r_hits <= 1.0) & (r_hits >= r_scale))
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX
    if targetId:
        assert np.all(hits["objectId"] == 1)
    else:
        assert np.all((hits["objectId"] == 1) | (hits["objectId"] == 2))


# TODO: Maybe a bit excessive on the amount of configs...
@pytest.mark.parametrize("guide", [True, False])
@pytest.mark.parametrize("targetId", [True, False])
@pytest.mark.parametrize("inlineVolumeModels", [True, False])
@pytest.mark.parametrize("disableDirect", [True, False])
@pytest.mark.parametrize("disableScattering", [True, False])
def test_SceneBackwardTargetTracer(
    guide: bool,
    targetId: bool,
    inlineVolumeModels: bool,
    disableDirect: bool,
    disableScattering: bool,
) -> None:
    if not isRayTracingEnabled():
        pytest.skip("ray tracing is not supported")

    N = 32 * 256
    MAX_LENGTH = 10
    T0, T1 = 10.0 * u.ns, 20.0 * u.ns
    T_MAX = 1.0 * u.us
    capacity = 48 * 256
    cam_pos = (-1.0, -7.0, -10.0) * u.m

    # create materials
    attenuate = Attenuating()
    absorbing = Attenuating(absorb=True)
    water = PureWaterModel().createMedium(physicModel=attenuate)
    water2 = PureWaterModel().createMedium(physicModel=absorbing, name="water2")
    glass = BK7Model().createMedium()
    absorber = AbsorbingSurface()
    border = BorderSurface()
    dielectric = DielectricSurface()
    mat = Material("mat", glass, water, dielectric, flags=("RL", "B"))
    matAbs = Material("abs", None, water, absorber)
    matVol = Material("vol", water if inlineVolumeModels else water2, water, border)
    matStore = MaterialStore([mat, matAbs, matVol])
    waterIdx = matStore.media["water"]
    if inlineVolumeModels:
        assert len(matStore.volumeModels) == 2
    else:
        assert len(matStore.volumeModels) == 3
    # create scene
    # it will consist of 6 spheres arranged in an octahedron
    store = MeshStore({"cube": "assets/cube.ply", "sphere": "assets/sphere.stl"})
    r, d = 40.0 * u.m, 5.0 * u.m
    a = 60.0 * u.m  # >= (2r+d) / sqrt(2) = min dist between spheres
    r_scale = 0.99547149974733 * u.m  # radius of inscribed sphere (icosphere)
    r_insc = r * r_scale
    x, y, z = 10.0, 5.0, -5.0
    _t = lambda dx, dy, dz: Transform.TRS(scale=r, translate=(x + dx, y + dy, z + dz))
    instances = [
        store.createInstance("sphere", "mat", _t(0, 0, a), detectorId=1),
        store.createInstance("sphere", "mat", _t(0, 0, -a), detectorId=2),
        store.createInstance("sphere", "abs", _t(0, a, 0)),
        store.createInstance("sphere", "abs", _t(0, -a, 0)),
        store.createInstance("sphere", "vol", _t(a, 0, 0)),
        store.createInstance("sphere", "vol", _t(-a, 0, 0)),
    ]
    scene = Scene(instances, matStore)
    targetGuide = SphereTargetGuide(position=(-x, -y, z + r + d), radius=r)

    # calculate min time
    if targetId:
        # detector #1 is farther away
        target_pos = (x, y, z + a) * u.m  # detector #1
        d_min = np.sqrt(np.square(np.subtract(target_pos, cam_pos)).sum(-1)) - r
    else:
        target_pos = (x, y, z - a) * u.m  # detector #2
        d_min = np.sqrt(np.square(np.subtract(target_pos, cam_pos)).sum(-1)) - r
    vg = theia.material.getPropertySamples(water, "group_velocity")
    assert vg is not None
    v_max = np.max(vg)
    t_min = d_min / v_max + T0

    # create pipeline
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xC01DC0FFEE)
    photons = UniformWavelengthSource()
    camera = PointCamera(mediumIdx=waterIdx, position=cam_pos, timeDelta=T0)
    recorder = HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneBackwardTargetTracer(
        N,
        ray,
        camera,
        photons,
        recorder,
        rng,
        scene,
        capacity=capacity,
        callback=stats,
        maxPathLength=MAX_LENGTH,
        targetId=1 if targetId else None,
        sampleCoefficient=0.0 if disableScattering else float("NaN"),
        maxTime=T_MAX,
        targetGuide=targetGuide if guide else None,
        disableDirectLighting=disableDirect,
    )
    # run pipeline
    pl.runPipeline(tracer.collectStages())
    hp.checkCurrentDeviceHealth()

    # check hits
    assert recorder.queue is not None
    hits = recorder.queue.view(0)

    assert hits.count > 0
    assert hits.count <= tracer.maxHits
    hits = hits[: hits.count]

    # check config via stats
    assert stats.created == N
    assert stats.mismatch == 0
    assert stats.absorbed > 0
    assert stats.volume > 0
    if disableScattering:
        assert stats.scattered == 0
    else:
        assert stats.scattered > 0

    r_hits = np.sqrt(np.square(hits["position"]).sum(-1))
    assert np.all((r_hits <= 1.0) & (r_hits >= r_scale))
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    assert np.min(hits["time"]) >= t_min
    assert np.max(hits["time"]) <= T_MAX
    if targetId:
        assert np.all(hits["objectId"] == 1)
    else:
        assert np.all((hits["objectId"] == 1) | (hits["objectId"] == 2))
