import pytest

import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl
from hephaistos.queue import dumpQueue

import theia
import theia.camera
import theia.estimator
import theia.light
import theia.material
import theia.random
import theia.scene
import theia.trace
import theia.units as u

from theia.scene import Transform

"""
The goal of this test is to check SceneTracer is conserving energy by tracing a
scene where we expect no light to escape.
This allows us to build a chain of trust: After testing this, we can use it to
check other tracers.
"""


class MediumModel(
    theia.material.DispersionFreeMedium,
    theia.material.HenyeyGreensteinPhaseFunction,
    theia.material.KokhanovskyOceanWaterPhaseMatrix,
    theia.material.MediumModel,
):
    def __init__(self, a, s, g) -> None:
        theia.material.DispersionFreeMedium.__init__(
            self, n=1.33, ng=1.33, mu_a=a, mu_s=s
        )
        theia.material.HenyeyGreensteinPhaseFunction.__init__(self, g)
        theia.material.KokhanovskyOceanWaterPhaseMatrix.__init__(
            self, p90=0.66, theta0=0.25, alpha=4.0, xi=25.6
        )

    ModelName = "homogenous"


@pytest.mark.parametrize(
    "mu_a,mu_s,g,polarized",
    [
        (0.0, 0.005, 0.0, False),
        (0.05, 0.01, 0.0, False),
        (0.05, 0.01, -0.9, False),
        (0.05, 0.01, 0.9, False),
        (0.0, 0.005, 0.0, True),
        (0.05, 0.01, -0.9, True),
    ],
)
def test_SceneForwardTracer_GroundTruth(
    mu_a: float, mu_s: float, g: float, polarized: bool
) -> None:
    """
    Ground Truth Test:

    This test we can check against an analytic solution to verify this algorithm.
    After that, we can use this config to test other simulations.

    Scenario:
    Sphere detector filled with scattering medium, in center spherical light
    source.
    """
    if not hp.isRaytracingEnabled():
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
    scatter_coef = 0.05
    maxTime = float("inf")
    # simulation settings
    batch_size = 2 * 1024 * 1024
    n_batches = 40

    # create materials
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium()
    material = theia.material.Material("det", medium, None, flags="DB")
    matStore = theia.material.MaterialStore([material])
    # load meshes
    meshStore = theia.scene.MeshStore({"sphere": "assets/sphere.stl"})

    # create scene
    targets = [theia.scene.SphereBBox(position, radius)]
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = theia.scene.Scene(
        [target],
        matStore.material,
        medium=matStore.media["homogenous"],
        targets=targets,
    )
    # create light (delta pulse)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=position, timeRange=(t0, t0), budget=budget
    )
    # create tracer
    # rng = theia.random.SobolQRNG(seed=0xC0FFEE)
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    recorder = theia.estimator.HitRecorder(polarized=polarized)
    tracer = theia.trace.SceneForwardTracer(
        batch_size,
        light,
        photons,
        recorder,
        rng,
        scene=scene,
        maxPathLength=max_length,
        disableTargetSampling=True,  # We're inside so that wont work anyway
        scatterCoefficient=scatter_coef,
        maxTime=maxTime,
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    batches = []

    def process(config: int, task: int) -> None:
        batch = dumpQueue(recorder.view(config))
        batches.append(batch)

    pipeline = pl.Pipeline([rng, photons, light, tracer, recorder])
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create batches
    tasks = [
        {},
    ] * n_batches  # rng advances on its own, so we dont have to update any params
    scheduler.schedule(tasks)
    scheduler.wait()

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

    # if polarized: check stokes vector is valid
    if polarized:
        stokes = np.concatenate([b["stokes"] for b in batches])
        assert np.abs(stokes[:, 0] - 1.0).max() < 1e-5
        assert stokes[:, 1:].max() <= 1.0
        assert stokes[:, 1:].min() >= -1.0
        assert np.all(np.square(stokes[:, 1:]).sum(-1) <= 1.0)


@pytest.mark.parametrize(
    "mu_a,mu_s,g,polarized",
    [
        (0.0, 0.01, 0.0, False),
        (0.05, 0.005, 0.5, False),
        (0.0, 0.01, 0.0, True),
        (0.05, 0.005, -0.5, True),
    ],
)
def test_SceneForwardTracer_Crosscheck(
    mu_a: float, mu_s: float, g: float, polarized: bool
) -> None:
    """
    Here we test SceneForwardTracer's target sampling.
    Spherical light source with spherical target.
    Use GroundTruth to check against.
    """
    if not hp.isRaytracingEnabled():
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
    medium = model.createMedium()
    material = theia.material.Material("det", None, medium, flags="DB")
    matStore = theia.material.MaterialStore([material])
    # load meshes
    meshStore = theia.scene.MeshStore({"sphere": "assets/sphere.stl"})

    # create scene
    targets = [theia.scene.SphereBBox(position, radius)]
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = theia.scene.Scene(
        [target],
        matStore.material,
        medium=matStore.media["homogenous"],
        targets=targets,
    )
    # create light (delta pulse)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=light_pos, timeRange=(t0, t0), budget=budget
    )

    # Calculate ground truth

    # create tracer
    # rng = theia.random.SobolQRNG(seed=0xC0FFEE)
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    value = theia.estimator.UniformValueResponse()
    response = theia.estimator.HistogramHitResponse(
        value, nBins=n_bins, binSize=bin_size, t0=bin_t0, normalization=1 / batch_size
    )
    tracer = theia.trace.SceneForwardTracer(
        batch_size,
        light,
        photons,
        response,
        rng,
        scene=scene,
        maxPathLength=2 * max_length,  # higher variance -> more samples
        disableTargetSampling=True,
        scatterCoefficient=scatter_coef,
        maxTime=maxTime,
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    hists = []

    def process(config: int, task: int) -> None:
        hists.append(response.result(config).copy())

    pipeline = pl.Pipeline([rng, photons, light, tracer, response])
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [
        {},
    ] * n_batches  # rng advances on its own, so no updates
    scheduler.schedule(tasks)
    scheduler.wait()
    # combine histograms
    truth_hist = np.mean(hists, 0)
    truth = truth_hist.sum()

    # create estimate

    # create tracer
    value = theia.estimator.UniformValueResponse()
    response = theia.estimator.HistogramHitResponse(
        value,
        nBins=n_bins,
        binSize=bin_size,
        t0=bin_t0,
        normalization=1 / batch_size,
    )
    tracer = theia.trace.SceneForwardTracer(
        batch_size,
        light,
        photons,
        response,
        rng,
        scene=scene,
        maxPathLength=max_length,
        disableTargetSampling=False,
        scatterCoefficient=scatter_coef,
        maxTime=maxTime,
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    hists = []

    def process(config: int, task: int) -> None:
        hists.append(response.result(config).copy())

    pipeline = pl.Pipeline([rng, photons, light, tracer, response])
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [
        {},
    ] * n_batches  # rng advances on its own, so no updates
    scheduler.schedule(tasks)
    scheduler.wait()
    # combine histograms
    hist = np.mean(hists, 0)
    estimate = hist.sum()

    # check estimate
    assert truth > 0.0  # just to be safe
    assert abs(estimate / truth - 1.0) < 0.02
    # Compare early part of light curves
    # The ground truth algorithm suffers from high variance especially at later
    # times making a test there pointless.
    # Use log10 to compare curves.
    log_err = None
    with np.errstate(divide="ignore", invalid="ignore"):
        log_err = (np.log10(hist) - np.log10(truth_hist)) / np.log10(hist)
    log_err = np.nan_to_num(log_err, nan=0.0)
    assert np.abs(log_err).mean() < 0.01


@pytest.mark.parametrize(
    "mu_a,mu_s,g,disableDirect,polarized,err",
    [
        (0.0, 0.005, 0.0, False, False, 5e-3),
        (0.0, 0.005, 0.0, True, False, 0.02),
        (0.05, 0.01, 0.0, False, False, 5e-3),
        (0.05, 0.01, 0.0, True, False, 5e-3),
        (0.05, 0.01, -0.9, False, False, 0.07),
        (0.05, 0.01, 0.9, True, False, 0.01),
        (0.0, 0.005, 0.0, False, True, 5e-3),
        (0.0, 0.005, 0.0, True, True, 0.02),
        (0.05, 0.01, -0.9, False, True, 0.07),
    ],
)
def test_SceneBackwardTracer(
    mu_a: float, mu_s: float, g: float, disableDirect: bool, polarized: bool, err: float
):
    """Similar to SceneForwardTracer_GroundTruth, but use backward tracer instead"""
    if not hp.isRaytracingEnabled():
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
    scatter_coef = 0.05
    maxTime = float("inf")
    # simulation settings
    batch_size = 2 * 1024 * 1024
    n_batches = 10

    # create materials
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium()
    material = theia.material.Material("det", medium, None, flags="DB")
    matStore = theia.material.MaterialStore([material])
    # load meshes
    meshStore = theia.scene.MeshStore({"sphere": "assets/sphere.stl"})
    # the mesh is not really a sphere
    # to prevent all camera rays to be produced outside, we need to scale camera
    r_scale = 0.99547149974733 * u.m  # radius of inscribed sphere (icosphere)
    r_insc = radius * r_scale

    # create scene
    targets = [theia.scene.SphereBBox(position, radius)]
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = theia.scene.Scene(
        [target],
        matStore.material,
        medium=matStore.media["homogenous"],
        targets=targets,
    )

    # create light (delta pulse)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=position, timeRange=(t0, t0), budget=budget
    )
    # create camera
    camera = theia.camera.SphereCameraRaySource(
        position=position,
        radius=-r_insc,
    )
    # create tracer
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    recorder = theia.estimator.HitRecorder(polarized=polarized)
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneBackwardTracer(
        batch_size,
        light,
        camera,
        photons,
        recorder,
        rng,
        scene,
        callback=stats,
        maxPathLength=max_length,
        maxTime=maxTime,
        scatterCoefficient=scatter_coef,
        polarized=polarized,
        disableDirectLighting=disableDirect,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    total = 0
    stokes = []

    def process(config: int, task: int) -> None:
        nonlocal total
        result = recorder.view(config)
        result = result[: result.count]
        # undo attenuation
        vg = 1.0 / model.ng * u.c
        d = vg * (result["time"] - t0)
        value0 = result["contrib"] * np.exp(mu_a * d)
        # add to sum
        total += value0.sum()

        # copy stokes vector if polarized
        if polarized:
            stokes.append(result["stokes"].copy())

    pipeline = pl.Pipeline([rng, photons, light, camera, tracer, recorder])
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create batches
    tasks = [
        {},
    ] * n_batches  # rng advances on its own, so we dont have to update any params
    scheduler.schedule(tasks)
    scheduler.wait()

    # calculate direct contribution without attenuation
    directContrib = budget * np.exp(-mu_s * r_insc)
    # calculate expected contribution
    contrib = budget - directContrib if disableDirect else budget

    # check for energy conservation
    estimate = total / (batch_size * n_batches)
    assert np.abs(estimate / contrib - 1.0) < err
    # assert estimate < contrib  # biased by ignoring longer paths, i.e. missing energy

    # if polarized: check stokes vector is valid
    if polarized:
        stokes = np.concatenate(stokes)
        assert np.abs(stokes[:, 0] - 1.0).max() < 1e-5
        assert stokes[:, 1:].max() <= 1.0
        assert stokes[:, 1:].min() >= -1.0
        assert np.all(np.square(stokes[:, 1:]).sum(-1) <= 1.0)


@pytest.mark.parametrize(
    "mu_a,mu_s,g,sampleTarget,polarized",
    [
        (0.0, 0.01, 0.0, False, False),
        (0.05, 0.005, 0.0, False, False),
        (0.0, 0.01, 0.0, False, True),
        (0.05, 0.005, 0.0, False, True),
        (0.0, 0.01, 0.5, True, False),
        (0.05, 0.005, 0.5, True, False),
        (0.0, 0.01, -0.9, True, True),
        (0.05, 0.005, -0.9, True, True),
    ],
)
def test_VolumeForwardTracer_Crosscheck(
    mu_a: float, mu_s: float, g: float, sampleTarget: bool, polarized: bool
) -> None:
    """
    Spherical light source with spherical target.
    Use GroundTruth to check against.
    """
    if not hp.isRaytracingEnabled():
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
    max_length = 15
    scatter_coef = 2.0 * mu_s  # 0.02
    maxTime = 500.0  # limit as ground truth suffers from high variance at late times
    # simulation settings
    batch_size = 1 * 1024 * 1024
    n_batches = 20
    # binning config
    bin_t0 = 0.0
    bin_size = 20.0
    n_bins = 25

    # create materials
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium()
    material = theia.material.Material("det", None, medium, flags="DB")
    matStore = theia.material.MaterialStore([material])
    # load meshes
    meshStore = theia.scene.MeshStore({"sphere": "assets/sphere.stl"})

    # create scene
    targets = [theia.scene.SphereBBox(position, radius)]
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = theia.scene.Scene(
        [target],
        matStore.material,
        medium=matStore.media["homogenous"],
        targets=targets,
    )
    # create light (delta pulse)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=light_pos, timeRange=(t0, t0), budget=budget
    )

    # Calculate ground truth

    # create tracer
    # rng = theia.random.SobolQRNG(seed=0xC0FFEE)
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    response = theia.estimator.HistogramHitResponse(
        theia.estimator.UniformValueResponse(),
        nBins=n_bins,
        binSize=bin_size,
        t0=bin_t0,
        normalization=1 / batch_size,
    )
    tracer = theia.trace.SceneForwardTracer(
        batch_size,
        light,
        photons,
        response,
        rng,
        scene=scene,
        maxPathLength=max_length,
        disableTargetSampling=True,
        scatterCoefficient=scatter_coef,
        maxTime=maxTime,
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    hists = []

    def process(config: int, task: int) -> None:
        hists.append(response.result(config).copy())

    pipeline = pl.Pipeline([rng, photons, light, tracer, response])
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [
        {},
    ] * n_batches  # rng advances on its own, so no updates
    scheduler.schedule(tasks)
    scheduler.wait()
    # combine histograms
    truth_hist = np.mean(hists, 0)
    truth = truth_hist.sum()

    # create estimate

    # create tracer
    response = theia.estimator.HistogramHitResponse(
        theia.estimator.UniformValueResponse(),
        nBins=n_bins,
        binSize=bin_size,
        t0=bin_t0,
        normalization=1 / batch_size,
    )
    tracer = theia.trace.VolumeForwardTracer(
        batch_size,
        light,
        photons,
        response,
        rng,
        target=targets[0],
        nScattering=max_length,
        disableTargetSampling=not sampleTarget,
        scatterCoefficient=scatter_coef,
        maxTime=maxTime,
        polarized=polarized,
        medium=matStore.media["homogenous"],
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    hists = []

    def process(config: int, task: int) -> None:
        hists.append(response.result(config).copy())

    pipeline = pl.Pipeline([rng, photons, light, tracer, response])
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [
        {},
    ] * n_batches  # rng advances on its own, so no updates
    scheduler.schedule(tasks)
    scheduler.wait()
    # combine histograms
    hist = np.mean(hists, 0)
    estimate = hist.sum()

    # check estimate
    thres = 0.02 if sampleTarget else 0.08  # give more slack if not target sampling
    assert abs(estimate / truth - 1.0) < thres
    # Compare early part of light curves
    # The ground truth algorithm suffers from high variance especially at later
    # times making a test there pointless.
    # Use log10 to compare curves.
    log_err = None
    with np.errstate(divide="ignore", invalid="ignore"):
        log_err = (np.log10(hist) - np.log10(truth_hist)) / np.log10(hist)
    log_err = np.nan_to_num(log_err, nan=0.0)
    thres = 0.05 if sampleTarget else 0.12
    assert np.abs(log_err).mean() < thres


@pytest.mark.parametrize(
    "mu_a,mu_s,g,disableDirect,polarized",
    [
        (0.0, 0.01, 0.0, False, False),
        (0.05, 0.005, 0.0, False, False),
        (0.0, 0.01, 0.0, False, True),
        (0.05, 0.005, 0.0, True, True),
        (0.05, 0.005, 0.5, True, True),
        (0.01, 0.01, -0.5, False, True),
    ],
)
def test_VolumeBackwardTracer_Crosscheck(
    mu_a: float, mu_s: float, g: float, disableDirect: bool, polarized: bool
):
    """
    Here we have to use a crosscheck as we cannot limit our rays to the a sphere
    with the volume tracer.
    """
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
    t0 = 30.0 * u.ns
    lam = 400.0 * u.nm  # doesn't really matter
    # tracer settings
    max_length = 15
    scatter_coef = 2.0 * mu_s  # 0.02
    maxTime = 1000.0  # limit as ground truth suffers from high variance at late times
    # simulation settings
    batch_size = 1 * 1024 * 1024
    n_batches = 20
    # binning config
    bin_t0 = 0.0
    bin_size = 10.0
    n_bins = 100

    # create materials
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium()
    mat = theia.material.Material("abs", None, medium, flags="B")
    matStore = theia.material.MaterialStore([mat], media=[medium])
    # create empty scene
    meshStore = theia.scene.MeshStore({"sphere": "assets/sphere.stl"})
    t = theia.scene.Transform.Scale(0, 0, 0).translate(1e9, 1e9, 1e9)  # move away
    c = meshStore.createInstance("sphere", "abs", t)
    scene = theia.scene.Scene(
        [c], matStore.material, medium=matStore.media["homogenous"]
    )
    # create light and camera
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=lightPos, timeRange=(t0, t0), budget=lightBudget
    )
    camera = theia.camera.FlatCameraRaySource(
        width=width,
        length=length,
        position=camPos,
        direction=camDir,
        up=camUp,
    )

    # create ground truth

    # create tracer
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    response = theia.estimator.HistogramHitResponse(
        theia.estimator.UniformValueResponse(),
        nBins=n_bins,
        binSize=bin_size,
        t0=bin_t0,
        normalization=1 / batch_size,
    )
    tracer = theia.trace.SceneBackwardTracer(
        batch_size,
        light,
        camera,
        photons,
        response,
        rng,
        scene,
        maxPathLength=max_length,
        maxTime=maxTime,
        scatterCoefficient=scatter_coef,
        polarized=polarized,
        disableDirectLighting=disableDirect,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline # scheduler
    hists = []

    def process(config: int, task: int) -> None:
        hists.append(response.result(config).copy())

    pipeline = pl.Pipeline([rng, photons, light, camera, tracer, response])
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [
        {},
    ] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # combine histograms
    truth_hist = np.mean(hists, 0)
    truth = truth_hist.sum()

    # create estimate

    # create tracer
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    response = theia.estimator.HistogramHitResponse(
        theia.estimator.UniformValueResponse(),
        nBins=n_bins,
        binSize=bin_size,
        t0=bin_t0,
        normalization=1 / batch_size,
    )
    tracer = theia.trace.VolumeBackwardTracer(
        batch_size,
        light,
        camera,
        photons,
        response,
        rng,
        medium=matStore.media["homogenous"],
        nScattering=max_length,
        scatterCoefficient=scatter_coef,
        maxTime=maxTime,
        polarized=polarized,
        disableDirectLighting=disableDirect,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    hists = []

    def process(config: int, task: int) -> None:
        hists.append(response.result(config).copy())

    pipeline = pl.Pipeline([rng, photons, light, tracer, response])
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [
        {},
    ] * n_batches  # rng advances on its own, so no updates
    scheduler.schedule(tasks)
    scheduler.wait()
    # combine histograms
    hist = np.mean(hists, 0)
    estimate = hist.sum()

    # check estimate
    assert truth != 0.0
    assert np.abs(estimate / truth - 1.0) < 5e-6
    # Compare early part of light curves
    # The ground truth algorithm suffers from high variance especially at later
    # times making a test there pointless.
    # Use log10 to compare curves.
    log_err = None
    with np.errstate(divide="ignore", invalid="ignore"):
        log_err = (np.log10(hist) - np.log10(truth_hist)) / np.log10(hist)
    log_err = np.nan_to_num(log_err, nan=0.0)
    # assert np.abs(log_err).mean() < binErr
    assert np.abs(log_err).max() < 5e-6


# Unfortunately, this tracer converges rather slowly needing a large amount of
# samples. To speed things up, we tune the number of batches and error for each
# test
@pytest.mark.parametrize(
    "mu_a,mu_s,g,polarized,n_batches,err",
    [
        (0.0, 0.005, 0.0, False, 200, 0.035),
        (0.05, 0.01, 0.0, True, 200, 0.025),
        (0.0, 0.02, -0.4, False, 200, 2e-3),
        (0.05, 0.01, 0.6, False, 250, 0.03),
    ],
)
def test_BidirectionalPathTracer(
    mu_a: float, mu_s: float, g: float, polarized: bool, n_batches: int, err: float
):
    """
    Here we have to work around the fact, that BidirectionalPathTracer does not
    sample direct and single scatter paths. Since we use SceneBackwardTracer for
    cross check we simply disable the former, but need a second tracer to
    estimate the difference from the second.
    """
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    # Scene settings
    position = (12.0, 15.0, 0.2) * u.m
    radius = 100.0 * u.m
    # Light settings
    budget = 1e9
    t0 = 10.0 * u.ns
    lam = 400.0 * u.nm  # doesn't really matter
    # tracer settings
    cam_length = 10
    light_length = 10
    scatter_coef = 0.05
    maxTime = float("inf")
    # simulation settings
    batch_size = 256 * 1024

    # create materials
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium()
    material = theia.material.Material("det", medium, None, flags="DB")
    matStore = theia.material.MaterialStore([material])
    # load meshes
    meshStore = theia.scene.MeshStore({"sphere": "assets/sphere.stl"})
    # the mesh is not really a sphere
    # to prevent all camera rays to be produced outside, we need to scale camera
    r_scale = 0.99547149974733 * u.m  # radius of inscribed sphere (icosphere)
    r_insc = radius * r_scale

    # create scene
    targets = [theia.scene.SphereBBox(position, radius)]
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = theia.scene.Scene(
        [target],
        matStore.material,
        medium=matStore.media["homogenous"],
        targets=targets,
    )

    # create light (delta pulse)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=position, timeRange=(t0, t0), budget=budget
    )
    # create camera
    camera = theia.camera.SphereCamera(position=position, radius=-r_insc)
    # create tracer
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    recorder = theia.estimator.HitRecorder(polarized=polarized)
    tracer = theia.trace.BidirectionalPathTracer(
        batch_size,
        light,
        camera,
        photons,
        recorder,
        rng,
        scene,
        lightPathLength=light_length,
        cameraPathLength=cam_length,
        scatterCoefficient=scatter_coef,
        maxTime=maxTime,
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    total = 0
    # stokes = []

    def process(config: int, task: int) -> None:
        nonlocal total
        result = recorder.view(config)
        result = result[: result.count]
        # undo attenuation
        vg = 1.0 / model.ng * u.c
        d = vg * (result["time"] - t0)
        value0 = result["contrib"] * np.exp(mu_a * d)
        # add to sum
        total += value0.sum()

        # copy stokes vector if polarized
        # if polarized:
        #     stokes.append(result["stokes"].copy())

    pipeline = pl.Pipeline([rng, photons, light, camera, tracer, recorder])
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create batches
    tasks = [
        {},
    ] * n_batches  # rng advances on its own, so we dont have to update any params
    scheduler.schedule(tasks)
    scheduler.wait()

    # estimate for single scatter contributions
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    recorder = theia.estimator.HitRecorder(polarized=polarized)
    tracer = theia.trace.SceneBackwardTracer(
        batch_size,
        light,
        camera,
        photons,
        recorder,
        rng,
        scene,
        maxPathLength=2,
        maxTime=maxTime,
        scatterCoefficient=scatter_coef,
        polarized=polarized,
        disableDirectLighting=False,
    )
    rng.autoAdvance = tracer.nRNGSamples

    singleTotal = 0

    def process(config: int, task: int) -> None:
        nonlocal singleTotal
        result = recorder.view(config)
        result = result[: result.count]
        # undo attenuation
        vg = 1.0 / model.ng * u.c
        d = vg * (result["time"] - t0)
        value0 = result["contrib"] * np.exp(mu_a * d)
        # add to sum
        singleTotal += value0.sum()

    pipeline = pl.Pipeline([rng, photons, light, camera, tracer, recorder])
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create batches
    tasks = [
        {},
    ] * 50  # rng advances on its own, so we dont have to update any params
    scheduler.schedule(tasks)
    scheduler.wait()

    # check for energy conservation
    singleEstimate = singleTotal / (batch_size * 50)
    estimate = total / (batch_size * n_batches)
    expected = budget - singleEstimate
    assert np.abs(estimate / expected - 1.0) < err

    # if polarized: check stokes vector is valid
    # if polarized:
    #     stokes = np.concatenate(stokes)
    #     assert np.abs(stokes[:, 0] - 1.0).max() < 1e-5
    #     assert stokes[:, 1:].max() <= 1.0
    #     assert stokes[:, 1:].min() >= -1.0
    #     assert np.all(np.square(stokes[:, 1:]).sum(-1) <= 1.0)


@pytest.mark.parametrize(
    "mu_a,mu_s,g,polarized",
    [
        (0.0, 0.005, 0.0, False),
        (0.05, 0.01, 0.0, False),
        (0.05, 0.01, -0.9, False),
        (0.05, 0.01, 0.9, False),
        (0.0, 0.005, 0.0, True),
        (0.05, 0.01, -0.9, True),
    ],
)
def test_DirectTracer(mu_a: float, mu_s: float, g: float, polarized: bool):
    """Similar to SceneForwardTracer_GroundTruth, except here we use DirectTracer"""

    # Scene settings
    position = (12.0, 15.0, 0.2) * u.m
    radius = 100.0 * u.m
    # Light settings
    budget = 1e9
    t0 = 10.0 * u.ns
    lam = 400.0 * u.nm  # doesn't really matter
    # simulation settings
    maxTime = float("inf")
    batch_size = 2 * 1024 * 1024
    n_batches = 10

    # create materials
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium()
    matStore = theia.material.MaterialStore([], media=[medium])

    # create light (delta pulse)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=position, timeRange=(t0, t0), budget=budget
    )
    # create camera
    camera = theia.camera.SphereCamera(position=position, radius=-radius)
    # create tracer
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    recorder = theia.estimator.HitRecorder(polarized=polarized)
    tracer = theia.trace.DirectLightTracer(
        batch_size,
        light,
        camera,
        photons,
        recorder,
        rng,
        maxTime=maxTime,
        medium=matStore.media["homogenous"],
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    batches = []

    def process(config: int, task: int) -> None:
        batch = dumpQueue(recorder.view(config))
        batches.append(batch)

    pipeline = pl.Pipeline([rng, photons, light, camera, tracer, recorder])
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create batches
    tasks = [
        {},
    ] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()

    # concat results
    time = np.concatenate([b["time"] for b in batches])
    value = np.concatenate([b["contrib"] for b in batches])

    # check expected arrival time
    vg = 1.0 / model.ng * u.c
    t = t0 + radius / vg
    assert np.allclose(time, t)

    # check for energy conservation
    estimate = value.sum() * tracer.normalization / n_batches
    expected = budget * np.exp(-(mu_a + mu_s) * radius)
    assert np.abs(estimate / expected - 1.0) < 5e-5

    # additional check since we have the data: uniform hits on sphere
    positions = np.concatenate([b["position"] for b in batches], axis=0)
    assert np.abs(positions.mean(0)).max() < 5e-3
    # assert np.abs(positions.var(0) - 1 / 3).max() < 0.1
    # TODO: Check why this ^^^ fails
    vars = np.vstack([b["position"].var(0) for b in batches])
    assert np.abs(vars - 1 / 3).max() < 0.01

    # if polarized: check stokes vector is valid
    if polarized:
        stokes = np.concatenate([b["stokes"] for b in batches])
        assert np.abs(stokes[:, 0] - 1.0).max() < 1e-5
        assert stokes[:, 1:].max() <= 1.0
        assert stokes[:, 1:].min() >= -1.0
        assert np.all(np.square(stokes[:, 1:]).sum(-1) <= 1.0)
