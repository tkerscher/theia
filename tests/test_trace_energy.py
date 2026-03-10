import pytest

import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl
from hephaistos.queue import dumpQueue

from theia.camera import PointCamera, SphereCamera
from theia.device import isRayTracingEnabled
from theia.light import ConstWavelengthSource, UniformWavelengthSource
from theia.light import SphericalLightSource
from theia.material import Material, MaterialStore
from theia.material import DispersionFreeMedium, HenyeyGreensteinPhaseFunction
from theia.ray import UnpolarizedRay
from theia.response import (
    HistogramHitResponse,
    HitRecorder,
    IntegratingHitResponse,
    UniformValueResponse,
)
from theia.random import PhiloxRNG
from theia.scene import MeshStore, Scene, Transform
from theia.surface import AbsorbingSurface, DielectricSurface
from theia.target import InnerSphereTarget, SphereTargetGuide
from theia.trace import (
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
        (0.05, 0.01, 0.9, False),
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
    max_length = 16
    maxTime = float("inf")
    # simulation settings
    batch_size = 2 * 1024 * 1024
    n_batches = 20

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
