import pytest

import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl
from hephaistos.queue import dumpQueue

from theia.camera import DiskCamera, FlatCamera, PointCamera, SphereCamera
from theia.device import isRayTracingEnabled
from theia.light import ConstWavelengthSource, UniformWavelengthSource
from theia.light import ConeLightSource, SphericalLightSource
from theia.material import Material, MaterialStore
from theia.property import TableProperty
from theia.model import (
    BK7Model,
    DispersionFreeMedium,
    HenyeyGreensteinPhaseFunction,
    PureWaterModel,
)
from theia.ray import UnpolarizedRay
from theia.response import (
    HistogramHitResponse,
    HitRecorder,
    IntegratingHitResponse,
    UniformValueResponse,
)
from theia.random import PhiloxRNG
from theia.scene import linkPortals, MeshStore, MultiScene, Scene, Transform
from theia.surface import (
    AbsorbingSurface,
    BorderSurface,
    DielectricSurface,
    LambertianReflectingSurface,
)
from theia.target import FlatTargetGuide, InnerSphereTarget, SphereTargetGuide
from theia.trace import (
    EventStatisticCallback,
    SceneBackwardTracer,
    SceneBackwardTargetTracer,
    SceneForwardTracer,
    VolumeBackwardTracer,
    VolumeForwardTracer,
)
from theia.volume import Attenuating
import theia.units as u

"""
The goal of this test is to check SceneTracer is conserving energy by tracing a
scene where we expect no light to escape.
This allows us to build a chain of trust: After testing this, we can use it to
check other tracers.
"""

pytestmark = pytest.mark.slow


class MediumModel(DispersionFreeMedium, HenyeyGreensteinPhaseFunction):
    def __init__(
        self, mu_a: float, mu_s: float, g: float, *, n: float = 1.33, ng: float = 1.33
    ) -> None:
        super().__init__(n=n, ng=ng, mu_a=mu_a, mu_s=mu_s, g=g, name="homogenous")


class UndoAttenuationProcessFn:
    """Scheduler callback summing up contributions after undoing absorption"""

    def __init__(self, recorder: HitRecorder, model: MediumModel, t0: float):
        self.recorder = recorder
        self.vg = 1.0 / model.ng * u.c
        self.mu_a = model.mu_a
        self.t0 = t0
        self.total = 0

    def __call__(self, config: int, batch: int, args) -> None:
        result = self.recorder.queue.view(config)  # type: ignore
        result = result[: result.count]
        # undo attenuation
        d = self.vg * (result["time"] - self.t0)
        value0 = result["contrib"] * np.exp(self.mu_a * d)
        # add to total
        self.total += value0.sum(dtype=np.float64)


@pytest.mark.parametrize(
    "mu_a,mu_s,g",
    [
        (0.0, 0.005, 0.0),
        (0.05, 0.01, 0.0),
        (0.05, 0.01, -0.9),
        (0.05, 0.01, 0.9),
    ],
)
def test_SceneForwardTracer_GroundTruth(mu_a: float, mu_s: float, g: float) -> None:
    """
    Ground Truth Test:

    This test we can check against an analytic solution to verify this algorithm.
    After that, we can use this config to test other simulations.

    Scenario:
    Sphere detector filled with scattering medium, in center spherical light
    source.
    """
    if not isRayTracingEnabled():
        pytest.skip("ray tracing is not supported")

    # Scene settings
    position = (12.0, 15.0, 0.2) * u.m
    radius = 100.0 * u.m
    # Light settings
    budget = 1e9
    t0 = 10.0 * u.ns
    lam = 400.0 * u.nm  # doesn't really matter
    # tracer settings
    max_length = 10
    # simulation settings
    batch_size = 2 * 1024 * 1024
    n_batches = 40

    # create materials
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium(physicModel=Attenuating())
    material = Material("det", medium, None, AbsorbingSurface(), flags="DB")
    matStore = MaterialStore([material])
    medIdx = matStore.media["homogenous"]

    # create scene
    meshStore = MeshStore({"sphere": "assets/sphere.stl"})
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = Scene([target], matStore)

    # create pipeline
    rng = PhiloxRNG(key=0xC0FFEE)
    ray = UnpolarizedRay()
    photons = UniformWavelengthSource(lambdaRange=(lam, lam))
    light = SphericalLightSource(
        photons, mediumIdx=medIdx, position=position, timeRange=(t0, t0), budget=budget
    )
    recorder = HitRecorder()
    tracer = SceneForwardTracer(
        batch_size,
        ray,
        light,
        recorder,
        rng,
        scene,
        maxPathLength=max_length,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    batches = []
    process = lambda c, b, a: batches.append(dumpQueue(recorder.queue.view(c)))
    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [{}] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()

    # concat results
    time = np.concatenate([b["time"] for b in batches])
    value = np.concatenate([b["contrib"] for b in batches])

    # undo attenuation
    vg = 1.0 / model.ng * u.c
    d = vg * (time - t0)
    value0 = value * np.exp(mu_a * d)

    # check for energy conservation
    estimate = value0.sum() / (batch_size * n_batches)
    assert np.abs(estimate / budget - 1.0) < 0.05
    assert estimate < budget  # biased by ignoring longer paths, i.e. missing energy

    # additional check since we have the data: uniform hits on sphere
    positions = np.concatenate([b["position"] for b in batches], axis=0)
    assert np.abs(positions.mean(0)).max() < 5e-3
    # assert np.abs(positions.var(0) - 1 / 3).max() < 0.1
    # TODO: Check why this ^^^ fails
    vars = np.vstack([b["position"].var(0) for b in batches])
    assert np.abs(vars - 1 / 3).max() < 0.01


@pytest.mark.parametrize(
    "mu_a,mu_s,g",
    [
        (0.0, 0.01, 0.0),
        (0.05, 0.005, 0.5),
        (0.05, 0.005, -0.5),
    ],
)
def test_SceneForwardTracer_particle(mu_a: float, mu_s: float, g: float):
    if not isRayTracingEnabled():
        pytest.skip("ray tracing is not supported")

    # Scene settings
    position = (12.0, 15.0, 0.2) * u.m
    radius = 100.0 * u.m
    # Light settings
    budget = 1e9
    t0 = 10.0 * u.ns
    lam = 400.0 * u.nm  # doesn't really matter
    # tracer settings
    max_length = 64
    # simulation settings
    batch_size = 2 * 1024 * 1024
    n_batches = 40

    # create materials
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium(physicModel=Attenuating())
    material = Material("det", medium, None, AbsorbingSurface(), flags="DB")
    matStore = MaterialStore([material])
    medIdx = matStore.media["homogenous"]

    # create scene
    meshStore = MeshStore({"sphere": "assets/sphere.stl"})
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = Scene([target], matStore)

    # calculate ground truth

    # create tracer
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xC0FFEE)
    photons = UniformWavelengthSource(lambdaRange=(lam, lam))
    light = SphericalLightSource(
        photons, mediumIdx=medIdx, position=position, timeRange=(t0, t0), budget=budget
    )
    response = IntegratingHitResponse(UniformValueResponse())
    tracer = SceneForwardTracer(
        batch_size,
        ray,
        light,
        response,
        rng,
        scene,
        maxPathLength=max_length,  # higher variance -> more samples
        # maxTime=maxTime,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    sums = []
    process = lambda c, b, a: sums.append(response.result(c)[0] / batch_size)
    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [{}] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()
    hp.checkCurrentDeviceHealth()
    # get estimate
    truth = np.array(sums).mean() / budget  # ratio detected
    assert truth > 0.0

    # create estimate

    # create pipeline
    ray = UnpolarizedRay(particle=True)
    rng = PhiloxRNG(key=0xC0FFEE)
    photons = UniformWavelengthSource(lambdaRange=(lam, lam))
    light = SphericalLightSource(
        photons,
        mediumIdx=medIdx,
        position=position,
        timeRange=(t0, t0),
        emitParticles=True,
    )
    response = IntegratingHitResponse(UniformValueResponse())
    tracer = SceneForwardTracer(
        batch_size,
        ray,
        light,
        response,
        rng,
        scene,
        maxPathLength=max_length,
        # maxTime=maxTime,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    counts = []
    process = lambda c, b, a: counts.append(response.result(c)[0])
    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [{}] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()

    # check estimate
    estimate = np.array(counts).mean() / batch_size  # ratio detected
    assert abs(estimate - truth) < 1e-5


@pytest.mark.parametrize(
    "mu_a,mu_s,g",
    [
        (0.0, 0.01, 0.0),
        (0.05, 0.005, 0.5),
        (0.05, 0.005, -0.5),
    ],
)
@pytest.mark.parametrize("sampleCoef", [float("NaN"), 0.01])
def test_SceneForwardTracer_CrossCheck(
    mu_a: float, mu_s: float, g: float, sampleCoef: float
) -> None:
    """
    Here we test SceneForwardTracer's target sampling.
    Spherical light source with spherical target.
    Use GroundTruth to check against.
    """
    if not isRayTracingEnabled():
        pytest.skip("ray tracing is not supported")

    # Scene settings
    position = (0.0, 0.0, 0.0) * u.m
    radius = 5.0 * u.m
    # Light settings
    light_pos = (-6.0, 0.0, 0.0) * u.m
    budget = 1e9
    t0 = 30.0 * u.ns
    lam = 400.0 * u.nm  # doesn't really matter
    # tracer settings
    max_length = 10
    scatter_coef = 0.01
    maxTime = 600.0  # limit as ground truth suffers from high variance at late times
    # simulation settings
    batch_size = 1 * 1024 * 1024
    n_batches = 100
    # binning config
    bin_t0 = 0.0
    bin_size = 20.0
    n_bins = 30

    # create materials
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium(physicModel=Attenuating())
    material = Material("det", None, medium, AbsorbingSurface(), flags="DB")
    matStore = MaterialStore([material])
    medIdx = matStore.media["homogenous"]

    # create scene
    meshStore = MeshStore({"sphere": "assets/sphere.stl"})
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = Scene([target], matStore)
    guide = SphereTargetGuide(position=position, radius=radius)
    # create light
    photons = UniformWavelengthSource(lambdaRange=(lam, lam))
    light = SphericalLightSource(
        photons, mediumIdx=medIdx, position=light_pos, timeRange=(t0, t0), budget=budget
    )

    # calculate ground truth

    # create tracer
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xC0FFEE)
    response = HistogramHitResponse(
        UniformValueResponse(),
        binCount=n_bins,
        binSize=bin_size,
        t0=bin_t0,
        normalization=1 / batch_size,
    )
    tracer = SceneForwardTracer(
        batch_size,
        ray,
        light,
        response,
        rng,
        scene,
        maxPathLength=8 * max_length,  # higher variance -> more samples
        # maxTime=maxTime,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    hists = []
    process = lambda c, b, a: hists.append(np.copy(response.result(c)))
    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [{}] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to free resources
    scheduler.destroy()
    # combine histograms
    truth_hist = np.mean(hists, 0)
    truth = truth_hist.sum()
    assert truth > 0.0  # fail early

    # create estimate

    # create pipeline
    response = HistogramHitResponse(
        UniformValueResponse(),
        binCount=n_bins,
        binSize=bin_size,
        t0=bin_t0,
        normalization=1 / batch_size,
    )
    tracer = SceneForwardTracer(
        batch_size,
        ray,
        light,
        response,
        rng,
        scene,
        maxPathLength=max_length,
        targetGuide=guide,
        sampleCoefficient=sampleCoef,
        # maxTime=maxTime
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    hists = []
    process = lambda c, b, a: hists.append(np.copy(response.result(c)))
    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [{}] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to free resources
    scheduler.destroy()
    # combine histogram
    hist = np.mean(hists, 0)
    estimate = hist.sum()

    hp.checkCurrentDeviceHealth()

    # check estimate
    assert abs(estimate / truth - 1.0) < 0.02
    # Compare early part of light curves
    # The ground truth algorithm suffers from high variance especially at later
    # times making a test there pointless.
    # Use log10 to compare curves.
    log_err = None
    with np.errstate(divide="ignore", invalid="ignore"):
        log_err = (np.log10(hist) - np.log10(truth_hist)) / np.log10(hist)
    log_err = np.nan_to_num(log_err, nan=0.0)
    assert np.abs(log_err).mean() < 0.02


@pytest.mark.parametrize(
    "mu_a,mu_s,g,direct,useTarget,err",
    [
        (0.0, 0.005, 0.0, True, True, 1e-4),
        (0.0, 0.005, 0.0, True, False, 1e-4),
        (0.0, 0.005, 0.0, False, True, 5e-4),
        (0.05, 0.01, 0.0, True, True, 5e-3),
        (0.05, 0.01, 0.0, False, True, 6e-3),
        (0.05, 0.01, -0.9, True, True, 0.015),
        (0.05, 0.01, 0.9, False, True, 1e-3),
        (0.05, 0.01, 0.9, False, False, 5e-4),
    ],
)
@pytest.mark.parametrize("sampleCoef", [float("NaN"), 0.01])
def test_VolumeForwardTracer(
    mu_a: float,
    mu_s: float,
    g: float,
    direct: bool,
    useTarget: bool,
    sampleCoef: float,
    err: float,
):
    """Spherical light source placed within a spherical target"""

    # Scene settings
    position = (12.0, 15.0, 0.2) * u.m
    radius = 100.0 * u.m
    # light settings
    budget = 1e9
    t0 = 10.0 * u.ns
    lam = 400.0 * u.nm
    # tracer settings
    max_length = 10
    maxTime = float("inf")
    # simulation settings
    batch_size = 512 * 1024
    n_batches = 20

    # create medium
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium(physicModel=Attenuating())
    store = MaterialStore([], media=[medium])
    medIdx = store.media["homogenous"]
    # create tracer
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xC0FFEE)
    photons = UniformWavelengthSource(lambdaRange=(lam, lam))
    light = SphericalLightSource(
        photons,
        mediumIdx=medIdx,
        position=position,
        timeRange=(t0, t0),
        budget=budget,
    )
    target = InnerSphereTarget(position=position, radius=radius)
    recorder = HitRecorder()
    tracer = VolumeForwardTracer(
        batch_size,
        ray,
        light,
        target,
        recorder,
        rng,
        store,
        volume="homogenous",
        maxPathLength=max_length,
        maxTime=maxTime,
        sampleCoefficient=sampleCoef,
        disableDirectLighting=not direct,
        disableTargetSampling=not useTarget,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    process = UndoAttenuationProcessFn(recorder, model, t0)
    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [{}] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()

    # check if everything went alright
    hp.checkCurrentDeviceHealth()

    # calculate direct contribution without attenuation
    directContrib = budget * np.exp(-mu_s * radius)
    # calculate expected contribution
    contrib = budget if direct else budget - directContrib
    # check for energy conservation
    estimate = process.total / (batch_size * n_batches)
    assert np.abs(estimate / contrib - 1.0) < err


@pytest.mark.parametrize(
    "mu_a,mu_s,g,disableDirect",
    [
        (0.0, 0.01, 0.0, False),
        (0.05, 0.005, 0.0, False),
        (0.05, 0.005, 0.5, True),
        (0.01, 0.01, -0.5, False),
    ],
)
@pytest.mark.parametrize("sampleCoef", [float("NaN"), 0.01])
def test_VolumeBackwardTracer(
    mu_a: float,
    mu_s: float,
    g: float,
    disableDirect: bool,
    sampleCoef: float,
):
    """Spherical light source placed within a spherical target"""

    # Scene settings
    position = (12.0, 15.0, 0.2) * u.m
    radius = 100.0 * u.m
    # light settings
    budget = 1e9
    t0 = 10.0 * u.ns
    lam = 400.0 * u.nm
    # tracer settings
    max_length = 64
    maxTime = float("inf")
    # simulation settings
    batch_size = 512 * 1024
    n_batches = 40

    # create medium
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium(physicModel=Attenuating())
    store = MaterialStore([], media=[medium])
    medIdx = store.media["homogenous"]
    # create tracer
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xC0FFEE)
    photons = UniformWavelengthSource(lambdaRange=(lam, lam))
    light = SphericalLightSource(
        photons,
        mediumIdx=medIdx,
        position=position,
        timeRange=(t0, t0),
        budget=budget,
    )
    camera = SphereCamera(mediumIdx=medIdx, position=position, radius=-radius)
    # make target a bit larger to not occlude the camera
    target = InnerSphereTarget(position=position, radius=radius * 1.001)
    recorder = HitRecorder()
    tracer = VolumeBackwardTracer(
        batch_size,
        ray,
        light,
        camera,
        photons,
        recorder,
        rng,
        store,
        volume="homogenous",
        maxPathLength=max_length,
        target=target,
        maxTime=maxTime,
        sampleCoefficient=sampleCoef,
        disableDirectLighting=disableDirect,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    process = UndoAttenuationProcessFn(recorder, model, t0)
    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [{}] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()
    hp.checkCurrentDeviceHealth()

    # calculate direct contribution without attenuation
    directContrib = budget * np.exp(-mu_s * radius)
    # calculate expected contribution
    contrib = budget - directContrib if disableDirect else budget
    # check for energy conservation
    estimate = process.total / (batch_size * n_batches)
    thres = 0.02 if disableDirect else 5e-3  # give more leeway
    assert np.abs(estimate / contrib - 1.0) < thres


@pytest.mark.parametrize(
    "mu_a,mu_s,g,disableDirect",
    [
        (0.0, 0.005, 0.0, False),
        (0.0, 0.005, 0.0, True),
        (0.05, 0.01, 0.0, False),
        (0.05, 0.01, 0.0, True),
        (0.05, 0.01, -0.9, False),
        # (0.05, 0.01, 0.9, False), # converges rather slowly
    ],
)
@pytest.mark.parametrize("sampleCoef", [float("NaN"), 0.02])
def test_SceneBackwardTracer(
    mu_a: float,
    mu_s: float,
    g: float,
    disableDirect: bool,
    sampleCoef: float,
):
    """Similar to SceneForwardTracer_GroundTruth, but use backward tracer instead"""
    if not isRayTracingEnabled():
        pytest.skip("Ray tracing is not supported")

    # Scene settings
    position = (12.0, 15.0, 0.2) * u.m
    radius = 50.0 * u.m
    # Light settings
    budget = 1e9
    t0 = 10.0 * u.ns
    lam = 400.0 * u.nm  # doesn't really matter
    # tracer settings
    max_length = 24
    maxTime = float("inf")
    # simulation settings
    batch_size = 1024 * 1024
    n_batches = 50

    # create materials
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium(physicModel=Attenuating())
    material = Material("det", medium, None, AbsorbingSurface(), flags="DB")
    matStore = MaterialStore([material])
    medIdx = matStore.media["homogenous"]

    # create scene
    meshStore = MeshStore({"sphere": "assets/sphere.stl"})
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = Scene([target], matStore)
    # the mesh is not really a sphere
    # to prevent all camera rays to be produced outside, we need to scale camera
    r_scale = 0.99547149974733 * u.m  # radius of inscribed sphere (icosphere)
    r_insc = radius * r_scale

    # create light (delta pulse)
    photons = UniformWavelengthSource(lambdaRange=(lam, lam))
    light = SphericalLightSource(
        photons,
        mediumIdx=medIdx,
        position=position,
        timeRange=(t0, t0),
        budget=budget,
    )
    # create camera
    camera = SphereCamera(mediumIdx=medIdx, position=position, radius=-r_insc)
    # create tracer
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xC0FFEE)
    recorder = HitRecorder()
    tracer = SceneBackwardTracer(
        batch_size,
        ray,
        light,
        camera,
        photons,
        recorder,
        rng,
        scene,
        maxPathLength=max_length,
        maxTime=maxTime,
        sampleCoefficient=sampleCoef,
        disableDirectLighting=disableDirect,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    process = UndoAttenuationProcessFn(recorder, model, t0)
    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [{}] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()
    hp.checkCurrentDeviceHealth()

    # calculate direct contribution without attenuation
    directContrib = budget * np.exp(-mu_s * r_insc)
    # calculate expected contribution
    contrib = budget - directContrib if disableDirect else budget

    # check for energy conservation
    estimate = process.total / (batch_size * n_batches)
    assert np.abs(estimate / contrib - 1.0) < 6e-3


@pytest.mark.parametrize("sampleCoef", [float("NaN"), 0.3])
def test_SceneBackwardTargetTracer(sampleCoef: float) -> None:
    if not isRayTracingEnabled():
        pytest.skip("ray tracing not supported")

    # scene settings
    position = (12.0, 15.0, 0.2) * u.m
    cam_pos = (10.0, 16.0, 0.0) * u.m  # offset camera
    radius = 10.0 * u.m
    radius_inner = 5.0 * u.m
    # the mesh is not really a sphere
    # to prevent all camera rays to be produced outside, we need to scale camera
    r_scale = 0.99547149974733 * u.m  # radius of inscribed sphere (icosphere)
    r_insc = radius * r_scale
    t0 = 10.0 * u.ns
    lam = 400.0 * u.nm  # doesn't really matter
    # tracer settings
    max_length = 40
    maxTime = float("inf")
    # simulation settings
    batch_size = 1024 * 1024
    n_batches = 20

    # create both media and material. Use Vacuum for outside
    # set mu_a=0 so we can skip the calculation to remove it
    n_inner, n_outer = 1.2, 1.8
    model = lambda n: MediumModel(0.0, 0.04, 0.9, n=n, ng=n)
    m_inner = model(n_inner).createMedium(name="inner", physicModel=Attenuating())
    m_outer = model(n_outer).createMedium(name="outer", physicModel=Attenuating())
    mat_det = Material("det", m_outer, None, AbsorbingSurface(), flags="BL")
    # need both reflection and transmission to recover all the energy
    mat_inner = Material("inner", m_inner, m_outer, DielectricSurface(), flags="TR")
    matStore = MaterialStore([mat_det, mat_inner])
    medIdx = matStore.media["inner"]

    # create scene
    meshStore = MeshStore({"sphere": "assets/sphere.stl"})
    t_inner = Transform.TRS(scale=radius_inner, translate=position)
    t_det = Transform.TRS(scale=radius, translate=position)
    inner = meshStore.createInstance("sphere", "inner", t_inner)
    det = meshStore.createInstance("sphere", "det", t_det, detectorId=1)
    scene = Scene([inner, det], matStore)

    # create tracer
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xC0FFEE)
    photons = ConstWavelengthSource(lam)
    camera = PointCamera(mediumIdx=medIdx, position=cam_pos, timeDelta=t0)
    response = HistogramHitResponse(
        UniformValueResponse(), binCount=400, binSize=100.0 * u.ns
    )
    tracer = SceneBackwardTargetTracer(
        batch_size,
        ray,
        camera,
        photons,
        response,
        rng,
        scene,
        maxPathLength=max_length,
        targetId=1,
        sampleCoefficient=sampleCoef,
        maxTime=maxTime,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    hists = []
    process = lambda c, b, a: hists.append(np.copy(response.result(c)))
    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    scheduler.schedule([{}] * n_batches)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()
    # combine histograms
    hist = np.mean(hists, 0)
    estimate = hist.sum()

    # check estimate
    expected = 4.0 * np.pi * (n_inner / n_outer) ** 2
    assert abs(estimate / expected - 1.0) < 0.002


@pytest.mark.parametrize("mis", [True, False])
def test_SceneForwardTracer_NonSpecular(mis: bool) -> None:
    """
    Checks the MIS implementation in forward tracer. Light is traced inside a
    perfectly reflecting and isotropic sphere, with a perfect absorber detector
    clipped into it. Since light can neither escape nor be absorbed, the detector
    should accumulate all the light, which is easy to check for.
    """
    if not isRayTracingEnabled():
        pytest.skip("ray tracing not supported")

    # scene settings
    lam = 600.0 * u.nm
    t0 = 20.0 * u.ns
    budget = 1e6
    position = (12.0, -5.0, 4.2) * u.m
    radius = 50.0 * u.m
    det_pos = (100.0, -5.0, 4.2) * u.m
    srf_pos = (50.0, -5.0, 4.2) * u.m
    srf_nrm = (-1.0, 0.0, 0.0)
    srf_up = (0.0, 1.0, 0.0)
    light_pos = (-24.0, 12.0, 18.0) * u.m
    # tracer settings
    max_length = 100
    maxTime = float("inf")
    batch_size = 1024 * 1024
    n_batches = 32

    # create materials
    medium = MediumModel(0.0, 0.0, 0.0).createMedium()  # transparent
    sphereProps = {"reflectivity": TableProperty.createConstTable(1.0)}
    matSphere = Material(
        "sphere",
        medium,
        medium,
        LambertianReflectingSurface(),
        flags="R",
        properties=sphereProps,
    )
    matDet = Material("det", medium, medium, AbsorbingSurface(), flags="D")
    matStore = MaterialStore([matSphere, matDet])
    medIdx = matStore.media["homogenous"]
    # create scene
    meshStore = MeshStore({"sphere": "assets/sphere.stl", "cube": "assets/cube.ply"})
    t_sphere = Transform.TRS(scale=radius, translate=position)
    t_det = Transform.TRS(scale=radius, translate=det_pos)
    sphere = meshStore.createInstance("sphere", "sphere", t_sphere)
    det = meshStore.createInstance("cube", "det", t_det, detectorId=1)
    scene = Scene([sphere, det], matStore)

    # create tracer
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xC0FFEE)
    photons = ConstWavelengthSource(lam)
    light = SphericalLightSource(
        photons, mediumIdx=medIdx, position=light_pos, timeRange=(t0, t0), budget=budget
    )
    if mis:
        guide = FlatTargetGuide(
            width=2 * radius,
            height=2 * radius,
            position=srf_pos,
            normal=srf_nrm,
            up=srf_up,
        )
    else:
        guide = None
    response = IntegratingHitResponse(UniformValueResponse())
    tracer = SceneForwardTracer(
        batch_size,
        ray,
        light,
        response,
        rng,
        scene,
        targetGuide=guide,
        maxPathLength=max_length,
        maxTime=maxTime,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    sums = []
    process = lambda c, b, a: sums.append(response.result(c)[0] / batch_size)
    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [{}] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()
    hp.checkCurrentDeviceHealth()

    # check result: we should get all the light back
    est = np.array(sums).mean()
    err = abs(est.item() / budget - 1.0)
    assert err < 5e-4


def test_SceneBackwardTracer_SurfaceNEE() -> None:
    if not isRayTracingEnabled():
        pytest.skip("ray tracing not supported")

    # scene settings
    lam = 600.0 * u.nm
    t0 = 20.0 * u.ns
    budget = 1e6
    position = (12.0, -5.0, 4.2) * u.m
    radius = 50.0 * u.m
    det_pos = (100.0, -5.0, 4.2) * u.m
    srf_pos = (49.99, -5.0, 4.2) * u.m  # slight offset to prevent self-intersection
    srf_nrm = (-1.0, 0.0, 0.0)
    srf_up = (0.0, 1.0, 0.0)
    light_pos = (-24.0, 12.0, 18.0) * u.m
    d = 100.0 - 12.0
    h = np.sqrt(radius**2 - (d - radius) ** 2)
    # tracer settings
    max_length = 256
    maxTime = float("inf")
    batch_size = 1024 * 1024
    n_batches = 24

    # create materials
    medium = MediumModel(0.0, 0.0, 0.0).createMedium()  # transparent
    sphereProps = {"reflectivity": TableProperty.createConstTable(1.0)}
    matSphere = Material(
        "sphere",
        medium,
        medium,
        LambertianReflectingSurface(),
        flags="R",
        properties=sphereProps,
    )
    matDet = Material("det", medium, medium, AbsorbingSurface(), flags="D")
    matStore = MaterialStore([matSphere, matDet])
    medIdx = matStore.media["homogenous"]
    # create scene
    meshStore = MeshStore({"sphere": "assets/sphere.stl", "cube": "assets/cube.ply"})
    t_sphere = Transform.TRS(scale=radius, translate=position)
    t_det = Transform.TRS(scale=radius, translate=det_pos)
    sphere = meshStore.createInstance("sphere", "sphere", t_sphere)
    det = meshStore.createInstance("cube", "det", t_det, detectorId=1)
    scene = Scene([sphere, det], matStore)

    # create tracer
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xC0FFEE)
    photons = ConstWavelengthSource(lam)
    light = SphericalLightSource(
        photons, mediumIdx=medIdx, position=light_pos, timeRange=(t0, t0), budget=budget
    )
    camera = DiskCamera(
        radius=h,
        position=srf_pos,
        direction=srf_nrm,
        up=srf_up,
        objectId=1,
        mediumIdx=medIdx,
    )
    response = IntegratingHitResponse(UniformValueResponse())
    tracer = SceneBackwardTracer(
        batch_size,
        ray,
        light,
        camera,
        photons,
        response,
        rng,
        scene,
        maxPathLength=max_length,
        maxTime=maxTime,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    sums = []
    process = lambda c, b, a: sums.append(response.result(c)[0] / batch_size)
    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [{}] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()
    hp.checkCurrentDeviceHealth()

    # check result: we should get all the light back
    est = np.array(sums).mean()
    err = abs(est.item() / budget - 1.0)
    # TODO: For whatever reason this one converges rather slowly
    assert err < 1e-3


def test_SceneForwardTracer_MultiScene() -> None:
    """
    Checks the SceneForwardTracer on a MultiScene against a flat Scene by tracing
    the very same geometry twice: once as a single flat scene, once split into
    sub-scenes entered through portals. Both must produce the same signal in
    each of the three detectors.

    Scene: a cone source in water shines along +x onto two glass spheres placed
    behind each other and rotated differently. Each sphere has a dielectric
    surface and holds an off-centre, rotated border box containing a small cube
    detector; a large cube detector sits behind both spheres.

    In the multi-scene version the sphere surfaces and the border boxes become
    portals, giving two sub-scenes - the sphere interior and the border box
    interior - each entered through TWO contexts, one per sphere. The sphere
    portal is the dielectric glass/water surface.

    Both sub-scenes are built ONCE, so the small detector exists as a single
    instance and tells the spheres apart through a per-context detector id, which
    the flat scene resolves with two separate instances instead. Comparing the
    detectors individually thus verifies the full portal mapping.
    """
    if not isRayTracingEnabled():
        pytest.skip("ray tracing not supported")

    # scene settings
    lam = 500.0 * u.nm
    budget = 1.0
    r_sphere = 0.50 * u.m  # the glass sphere, which is also the portal
    s_border = 0.15 * u.m  # half edge of the border box
    s_det = 0.07 * u.m  # half edge of the small detector cube
    box_off = (0.0, 0.10, 0.06) * u.m  # border box centre, sphere-local
    box_rot = (1.0, 1.0, 0.0, 25.0)  # border box rotation, sphere-local
    det_off = (0.05, 0.0, -0.03) * u.m  # detector centre, box-local
    sphere_pos = [(2.0, 0.0, 0.0) * u.m, (4.0, 0.0, 0.0) * u.m]
    sphere_rot = [(0.0, 0.0, 1.0, 35.0), (0.0, 1.0, 0.0, 70.0)]
    big_pos = (8.0, 0.0, 0.0) * u.m
    s_big = 2.0 * u.m
    light_pos = (0.0, 0.0, 0.0) * u.m
    cos_opening = 0.95
    # tracer settings
    batch_size = 64 * 1024
    n_batches = 16
    max_length = 24

    # create materials
    water = PureWaterModel().createMedium()
    glass = BK7Model().createMedium()
    matStore = MaterialStore(
        [
            Material("glass_portal", glass, water, DielectricSurface(), flags="PRT"),
            Material("border_glass", glass, glass, BorderSurface(), flags="P*"),
            Material("det_small", glass, glass, AbsorbingSurface(), flags="D"),
            Material("det_big", water, water, AbsorbingSurface(), flags="D"),
        ]
    )
    waterIdx = matStore.media["water"]

    # Placements: Sub-scene B is the interior of the border box, sub-scene A the 
    # interior of a sphere 
    store = MeshStore({"sphere": "assets/sphere.stl", "cube": "assets/cube.ply"})
    t_sphere = Transform.TRS(scale=r_sphere)  # sub-scene A
    t_borderBox = Transform.TRS(scale=s_border)  # sub-scene B
    t_det = Transform.TRS(scale=s_det, translate=det_off)  # sub-scene B
    t_big = Transform.TRS(scale=s_big, translate=big_pos)  # main scene
    # sub-scene A (sphere interior) -> world, one per sphere
    frameA = [
        Transform.TRS(rotate=rot, translate=pos)
        for rot, pos in zip(sphere_rot, sphere_pos)
    ]
    # sub-scene B (border box interior) -> sub-scene A, shared by both spheres
    frameB = Transform.TRS(rotate=box_rot, translate=box_off)

    # ---- flat reference scene: every piece placed in world coordinates ---- #
    flatInstances = []
    for k, f in enumerate(frameA):
        flatInstances += [
            store.createInstance("sphere", "glass_portal", f @ t_sphere),
            store.createInstance("cube", "border_glass", f @ frameB @ t_borderBox),
            # the two small detectors are distinct instances here
            store.createInstance(
                "cube", "det_small", f @ frameB @ t_det, detectorId=k
            ),
        ]
    flatInstances.append(store.createInstance("cube", "det_big", t_big, detectorId=2))
    flatScene = Scene(flatInstances, matStore)

    # ---- multi-scene: the same geometry behind portals -------------------- #
    # Both subscenes are entered with one context per sphere and therefore built
    # once.
    borderBoxB = store.createPortal("cube", "border_glass", t_borderBox, contextCount=2)
    detB = store.createInstance("cube", "det_small", t_det, detectorId=[0, 1])
    sphereA = store.createPortal("sphere", "glass_portal", t_sphere, contextCount=2)
    borderBoxA = store.createPortal(
        "cube", "border_glass", frameB @ t_borderBox, contextCount=2
    )
    # world scene: the two sphere portals plus the large detector, one context
    mainPortals = [
        store.createPortal("sphere", "glass_portal", f @ t_sphere) for f in frameA
    ]
    bigDet = store.createInstance("cube", "det_big", t_big, detectorId=2)

    # transition graph, all edges symmetric here
    for k, p in enumerate(mainPortals):
        linkPortals(p, sphereA, [(0, k)])  # world <-> A, context k per sphere
    linkPortals(borderBoxA, borderBoxB, [(0, 0), (1, 1)])  # A <-> B, per context

    multiScene = MultiScene(
        [[*mainPortals, bigDet], [sphereA, borderBoxA], [borderBoxB, detB]],
        matStore,
    )

    def run(multi: bool):
        """Traces the scene once per batch and returns the per-batch results"""
        rng = PhiloxRNG(key=0xC0FFEE)
        photons = ConstWavelengthSource(lam)
        light = ConeLightSource(
            photons,
            mediumIdx=waterIdx,
            position=light_pos,
            direction=(1.0, 0.0, 0.0),
            cosOpeningAngle=cos_opening,
            timeRange=(0.0, 0.0),
            budget=budget,
        )
        response = IntegratingHitResponse(UniformValueResponse(), detectorCount=3)
        stats = EventStatisticCallback()
        common = (batch_size, UnpolarizedRay(), light, response, rng)
        kwargs = dict(
            callback=stats,
            maxPathLength=max_length,
            sampleCoefficient=0.0,  # no volume scattering
        )
        scene = multiScene if multi else flatScene
        tracer = SceneForwardTracer(*common, scene, **kwargs)
        rng.autoAdvance = tracer.nRNGSamples

        results: list[np.ndarray] = []
        process = lambda c, b, a: results.append(
            np.array(response.result(c), dtype=np.float64)
        )
        scheduler = pl.PipelineScheduler(
            pl.Pipeline(tracer.collectStages()), processFn=process
        )
        scheduler.schedule([{}] * n_batches)
        scheduler.wait()
        scheduler.destroy()  # free resources before building the next tracer
        hp.checkCurrentDeviceHealth()
        return np.array(results), stats

    flat, flatStats = run(multi=False)
    multi, multiStats = run(multi=True)

    # a media inconsistency would silently kill photons and bias the comparison
    for stats in (flatStats, multiStats):
        assert stats.mismatch == 0
        assert stats.error == 0
    assert flatStats.created == multiStats.created  # same primary rays

    # we compare the means of both configurations
    mFlat, mMulti = flat.mean(0), multi.mean(0)
    seFlat = flat.std(0, ddof=1) / np.sqrt(n_batches)
    seMulti = multi.std(0, ddof=1) / np.sqrt(n_batches)
    sigma = np.hypot(seFlat, seMulti)

    # all three detectors must actually see light
    assert np.all(mFlat > 0.0)
    assert np.all(mMulti > 0.0)
    # The two spheres must give different signals - otherwise swapping the two
    # portal contexts would go unnoticed.
    assert abs(mFlat[0] - mFlat[1]) > 5.0 * np.hypot(seFlat[0], seFlat[1])
    # the actual check: same signal in every detector
    assert np.all(np.abs(mMulti - mFlat) < 5.0 * sigma)
