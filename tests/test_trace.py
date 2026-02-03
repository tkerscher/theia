import pytest

import hephaistos as hp
import hephaistos.pipeline as pl
import numpy as np

from itertools import product

from theia.camera import SphereCamera
from theia.light import PencilLightSource, SphericalLightSource, UniformWavelengthSource
from theia.material import MaterialStore, PureWaterModel, getPropertySamples, VACUUM_IDX
from theia.random import PhiloxRNG
from theia.ray import UnpolarizedRay
from theia.response import HitRecorder
from theia.target import SphereTarget
from theia.volume import Attenuating
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
    medium = theia.material.PureWaterModel().createMedium(physicModel=vol)
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
    assert codes.min() >= minCode and codes.max() <= maxCode


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
        photons,
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
