import pytest

import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl
from hephaistos.queue import dumpQueue

import theia
import theia.camera
import theia.response
import theia.light
import theia.material
import theia.random
import theia.scene
import theia.trace
import theia.units as u

from theia.scene import Transform
from theia.target import InnerSphereTarget, SphereTargetGuide

"""
The goal of this test is to check SceneTracer is conserving energy by tracing a
scene where we expect no light to escape.
This allows us to build a chain of trust: After testing this, we can use it to
check other tracers.
"""

pytestmark = pytest.mark.slow


class MediumModel(
    theia.material.DispersionFreeMedium,
    theia.material.HenyeyGreensteinPhaseFunction,
    theia.material.KokhanovskyOceanWaterPhaseMatrix,
    theia.material.MediumModel,
):
    def __init__(self, a, s, g, *, n=1.33, ng=1.33) -> None:
        theia.material.DispersionFreeMedium.__init__(self, n=n, ng=ng, mu_a=a, mu_s=s)
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
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = theia.scene.Scene(
        [target],
        matStore.material,
        medium=matStore.media["homogenous"],
    )
    # create light (delta pulse)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=position, timeRange=(t0, t0), budget=budget
    )
    # create tracer
    # rng = theia.random.SobolQRNG(seed=0xC0FFEE)
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    recorder = theia.response.HitRecorder(polarized=polarized)
    tracer = theia.trace.SceneForwardTracer(
        batch_size,
        light,
        photons,
        recorder,
        rng,
        scene=scene,
        maxPathLength=max_length,
        scatterCoefficient=scatter_coef,
        maxTime=maxTime,
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    batches = []

    def process(config: int, batch: int, args) -> None:
        batch = dumpQueue(recorder.queue.view(config))
        batches.append(batch)

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create batches
    tasks = [
        {},
    ] * n_batches  # rng advances on its own, so we dont have to update any params
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
    guide = SphereTargetGuide(position=position, radius=radius)
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = theia.scene.Scene(
        [target],
        matStore.material,
        medium=matStore.media["homogenous"],
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
    value = theia.response.UniformValueResponse()
    response = theia.response.HistogramHitResponse(
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
        scatterCoefficient=scatter_coef,
        maxTime=maxTime,
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    hists = []

    def process(config: int, batch: int, args) -> None:
        hists.append(response.result(config).copy())

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [
        {},
    ] * n_batches  # rng advances on its own, so no updates
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()
    # combine histograms
    truth_hist = np.mean(hists, 0)
    truth = truth_hist.sum()

    # create estimate

    # create tracer
    value = theia.response.UniformValueResponse()
    response = theia.response.HistogramHitResponse(
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
        targetGuide=guide,
        scatterCoefficient=scatter_coef,
        maxTime=maxTime,
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    hists = []

    def process(config: int, batch: int, args) -> None:
        hists.append(response.result(config).copy())

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    tasks = [{}] * n_batches  # rng advances on its own, so no updates
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()
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
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = theia.scene.Scene(
        [target],
        matStore.material,
        medium=matStore.media["homogenous"],
    )

    # create light (delta pulse)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=position, timeRange=(t0, t0), budget=budget
    )
    # create camera
    camera = theia.camera.SphereCamera(
        position=position,
        radius=-r_insc,
    )
    # create tracer
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    recorder = theia.response.HitRecorder(polarized=polarized)
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

    def process(config: int, batch: int, args) -> None:
        nonlocal total
        result = recorder.queue.view(config)
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

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create batches
    tasks = [
        {}
    ] * n_batches  # rng advances on its own, so we dont have to update any params
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()

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


@pytest.mark.parametrize("polarized", [True, False])
def test_SceneForwardTracer_MultiMedia(polarized: bool):
    """Tests SceneForwardTracer with multiple media

    Again, spherical light source inside a spherical detector, but this time
    the detector contains an outer and an inner shell of different media.
    """
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    # Scene settings
    position = (12.0, 15.0, 0.2) * u.m
    light_pos = (2.0, 5.0, 0.0) * u.m  # offset light
    radius = 100.0 * u.m
    radius_inner = 65.0 * u.m
    # the mesh is not really a sphere
    # to prevent all camera rays to be produced outside, we need to scale camera
    r_scale = 0.99547149974733 * u.m  # radius of inscribed sphere (icosphere)
    r_insc = radius * r_scale
    # Light settings
    budget = 1e9
    t0 = 10.0 * u.ns
    lam = 400.0 * u.nm  # doesn't really matter
    # tracer settings
    max_length = 50
    scatter_coef = 0.03
    maxTime = float("inf")
    # simulation settings
    batch_size = 2 * 1024 * 1024
    n_batches = 10

    # create both media and material. Use Vacuum for outside
    # set mu_a=0 so we can skip the calculation to remove it
    media_inner = MediumModel(0.0, 0.01, 0.9, n=1.8).createMedium(name="inner")
    media_outer = MediumModel(0.0, 0.04, 0.9, n=1.2).createMedium(name="outer")
    mat_det = theia.material.Material("det", media_outer, None, flags="DB")
    # need both reflection and transmission to recover all the energy
    mat_inner = theia.material.Material("inner", media_inner, media_outer, flags="TR")
    matStore = theia.material.MaterialStore([mat_det, mat_inner])

    # create scene
    meshStore = theia.scene.MeshStore({"sphere": "assets/sphere.stl"})
    t_inner = Transform.TRS(scale=radius_inner, translate=position)
    t_det = Transform.TRS(scale=radius, translate=position)
    inner = meshStore.createInstance("sphere", "inner", t_inner)
    det = meshStore.createInstance("sphere", "det", t_det)
    scene = theia.scene.Scene([inner, det], matStore.material)

    # create tracing pipeline
    photons = theia.light.ConstWavelengthSource(lam)
    light = theia.light.SphericalLightSource(
        position=light_pos, timeRange=(t0, t0), budget=budget
    )
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    response = theia.response.HistogramHitResponse(
        theia.response.UniformValueResponse(), nBins=400, binSize=100.0 * u.ns
    )
    tracer = theia.trace.SceneForwardTracer(
        batch_size,
        light,
        photons,
        response,
        rng,
        scene,
        maxPathLength=max_length,
        scatterCoefficient=scatter_coef,
        sourceMedium=matStore.media["inner"],
        maxTime=maxTime,
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    hists = []

    def process(config: int, batch: int, args) -> None:
        hists.append(response.result(config).copy())

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    scheduler.schedule([{} for _ in range(n_batches)])
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()
    # combine histograms
    hist = np.mean(hists, 0)
    estimate = hist.sum()

    # check estimate
    assert abs(estimate / budget - 1.0) < 0.0061


@pytest.mark.parametrize("polarized", [True, False])
def test_SceneBackwardTracer_MultiMedia(polarized: bool):
    """Tests SceneBackwardTracer with multiple media

    Again, spherical light source inside a spherical detector, but this time
    the detector contains an outer and an inner shell of different media.
    A problem with backward tracing is, that we cannot sample purely specular
    paths. We model the material so, that the only specular contribution is the
    straight connection from light to detector which can be estimated
    analytically. Results are compared between forward and backward tracer.
    """
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    # Scene settings
    position = (12.0, 15.0, 0.2) * u.m
    radius = 80.0 * u.m
    radius_inner = 50.0 * u.m
    # the mesh is not really a sphere
    # to prevent all camera rays to be produced outside, we need to scale camera
    r_scale = 0.99547149974733  # radius of inscribed sphere (icosphere)
    r_insc = radius * r_scale
    # Light settings
    budget = 1e9
    t0 = 10.0 * u.ns
    lam = 400.0 * u.nm  # doesn't really matter
    # tracer settings
    max_length = 20
    scatter_coef = 0.04
    maxTime = float("inf")
    # simulation settings
    batch_size = 1024 * 1024
    n_batches = 20

    # create media
    n_inner, n_outer = 1.8, 1.2
    mu_a = 0.005
    mu_s = 0.04
    media_inner = MediumModel(mu_a, mu_s, 0.9, n=n_inner).createMedium(name="inner")
    media_outer = MediumModel(mu_a, 0.0, 0.0, n=n_outer).createMedium(name="outer")
    # create materials
    mat_in_bwd = theia.material.Material(
        "inner_bwd", media_inner, media_outer, flags=("T", "B")
    )
    mat_out_bwd = theia.material.Material("det_bwd", media_outer, None, flags="DB")
    mat_in_fwd = theia.material.Material(
        "inner_fwd", media_inner, media_outer, flags=("B", "T")
    )
    mat_out_fwd = theia.material.Material("det_fwd", media_outer, None, flags="DB")
    matStore = theia.material.MaterialStore(
        [mat_in_bwd, mat_out_bwd, mat_in_fwd, mat_out_fwd]
    )

    # create scene
    meshStore = theia.scene.MeshStore({"sphere": "assets/sphere.stl"})
    t_inner = Transform.TRS(scale=radius_inner, translate=position)
    t_outer = Transform.TRS(scale=radius, translate=position)
    inner_bwd = meshStore.createInstance("sphere", "inner_bwd", t_inner)
    outer_bwd = meshStore.createInstance("sphere", "det_bwd", t_outer)
    inner_fwd = meshStore.createInstance("sphere", "inner_fwd", t_inner)
    outer_fwd = meshStore.createInstance("sphere", "det_fwd", t_outer)
    scene_bwd = theia.scene.Scene([inner_bwd, outer_bwd], matStore.material)
    scene_fwd = theia.scene.Scene([inner_fwd, outer_fwd], matStore.material)

    # backward tracer
    photons = theia.light.ConstWavelengthSource(lam)
    light = theia.light.SphericalLightSource(
        position=position, timeRange=(t0, t0), budget=budget
    )
    camera = theia.camera.SphereCamera(position=position, radius=-r_insc)
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    response = theia.response.HistogramHitResponse(
        theia.response.UniformValueResponse(), nBins=400, binSize=100.0 * u.ns
    )
    tracer = theia.trace.SceneBackwardTracer(
        batch_size,
        light,
        camera,
        photons,
        response,
        rng,
        scene_bwd,
        maxPathLength=max_length,
        scatterCoefficient=scatter_coef,
        medium=matStore.media["outer"],
        maxTime=maxTime,
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    hists = []

    def process(config: int, batch: int, args) -> None:
        hists.append(response.result(config).copy())

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    scheduler.schedule([{} for _ in range(n_batches)])
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()
    # combine histograms
    hist = np.mean(hists, 0)
    estimate_bwd = hist.sum()

    # trace forward
    photons = theia.light.ConstWavelengthSource(lam)
    light = theia.light.SphericalLightSource(
        position=position, timeRange=(t0, t0), budget=budget
    )
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    response = theia.response.HistogramHitResponse(
        theia.response.UniformValueResponse(), nBins=400, binSize=100.0 * u.ns
    )
    tracer = theia.trace.SceneForwardTracer(
        batch_size,
        light,
        photons,
        response,
        rng,
        scene_fwd,
        maxPathLength=max_length,
        scatterCoefficient=scatter_coef,
        sourceMedium=matStore.media["inner"],
        maxTime=maxTime,
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    hists = []

    def process(config: int, batch: int, args) -> None:
        hists.append(response.result(config).copy())

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    scheduler.schedule([{} for _ in range(n_batches)])
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()
    # combine histograms
    hist = np.mean(hists, 0)
    estimate_fwd = hist.sum()

    # contribution from purely specular path
    T = 1.0 - ((n_inner - n_outer) / (n_inner + n_outer)) ** 2
    spec_est = budget * T * np.exp(-mu_s * radius_inner) * np.exp(-mu_a * radius)

    # check estimate
    estimate = estimate_bwd + spec_est
    assert abs(estimate / estimate_fwd - 1.0) < 0.02  # maybe a bit large?


def test_SceneBackwardTargetTrace() -> None:
    """Tests SceneBackwardTargetTracer"""
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing not supported")

    # Scene settings
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
    scatter_coef = 0.3
    maxTime = float("inf")
    # simulation settings
    batch_size = 1024 * 1024
    n_batches = 20

    # create both media and material. Use Vacuum for outside
    # set mu_a=0 so we can skip the calculation to remove it
    n_inner, n_outer = 1.2, 1.8
    media_inner = MediumModel(0.0, 0.04, 0.9, n=n_inner).createMedium(name="inner")
    media_outer = MediumModel(0.0, 0.01, 0.9, n=n_outer).createMedium(name="outer")
    mat_det = theia.material.Material("det", media_outer, None, flags="BL")
    # need both reflection and transmission to recover all the energy
    mat_inner = theia.material.Material("inner", media_inner, media_outer, flags="TR")
    matStore = theia.material.MaterialStore([mat_det, mat_inner])

    # create scene
    meshStore = theia.scene.MeshStore({"sphere": "assets/sphere.stl"})
    t_inner = Transform.TRS(scale=radius_inner, translate=position)
    t_det = Transform.TRS(scale=radius, translate=position)
    inner = meshStore.createInstance("sphere", "inner", t_inner)
    det = meshStore.createInstance("sphere", "det", t_det, detectorId=1)
    scene = theia.scene.Scene([inner, det], matStore.material)

    # creating tracing pipeline
    photons = theia.light.ConstWavelengthSource(lam)
    camera = theia.camera.PointCamera(position=cam_pos, timeDelta=t0)
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    response = theia.response.HistogramHitResponse(
        theia.response.UniformValueResponse(), nBins=400, binSize=100.0 * u.ns
    )
    # stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.SceneBackwardTargetTracer(
        batch_size,
        camera,
        photons,
        response,
        rng,
        scene,
        maxPathLength=max_length,
        # callback=stats,
        scatterCoefficient=scatter_coef,
        targetId=1,
        medium=matStore.media["inner"],
        maxTime=maxTime,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    hists = []

    def process(config: int, batch: int, args) -> None:
        hists.append(response.result(config).copy())

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)
    # create batches
    scheduler.schedule([{} for _ in range(n_batches)])
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()
    # combine histograms
    hist = np.mean(hists, 0)
    estimate = hist.sum()

    # check estimate
    expected = 4.0 * np.pi * (n_inner / n_outer) ** 2
    assert abs(estimate / expected - 1.0) < 0.0005


@pytest.mark.parametrize(
    "mu_a,mu_s,mu_sample,g,disableDirect,sampleTarget,polarized,err",
    [
        (0.0, 0.005, 0.05, 0.0, False, True, False, 4.5e-4),
        (0.0, 0.005, 0.05, 0.0, False, False, False, 0.011),
        (0.0, 0.005, 0.05, 0.0, True, True, False, 0.011),
        (0.05, 0.01, 0.05, 0.0, False, True, False, 1.8e-3),
        (0.05, 0.01, 0.05, 0.0, True, True, False, 2.6e-3),
        (0.05, 0.01, 0.05, -0.9, False, True, False, 0.012),
        (0.05, 0.01, 0.05, 0.9, True, True, False, 2.8e-4),
        (0.05, 0.01, 0.05, 0.9, True, False, False, 2.9e-3),
        (0.05, 0.01, np.nan, 0.9, False, True, True, 2.9e-3),
        (0.0, 0.005, 0.05, 0.0, False, True, True, 5e-4),
        (0.0, 0.005, 0.05, 0.0, True, True, True, 1.2e-3),
        (0.05, 0.01, 0.05, -0.9, False, True, True, 0.012),
    ],
)
def test_VolumeForwardTracer(
    mu_a: float,
    mu_s: float,
    mu_sample: float | None,
    g: float,
    disableDirect: bool,
    sampleTarget: bool,
    polarized: bool,
    err: float,
) -> None:
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
    # batch_size = 2 * 1024 * 1024 # does not work on my laptop!?
    batch_size = 512 * 1024
    n_batches = 10

    # create medium
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium()
    store = theia.material.MaterialStore([], media=[medium])

    # create scene
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=position, timeRange=(t0, t0), budget=budget
    )
    target = InnerSphereTarget(position=position, radius=radius)
    # stats = theia.trace.EventStatisticCallback()
    recorder = theia.response.HitRecorder(polarized=polarized)
    tracer = theia.trace.VolumeForwardTracer(
        batch_size,
        light,
        target,
        photons,
        recorder,
        rng,
        # callback=stats,
        medium=store.media["homogenous"],
        maxTime=maxTime,
        nScattering=max_length,
        scatterCoefficient=mu_sample,
        polarized=polarized,
        disableDirectLighting=disableDirect,
        disableTargetSampling=not sampleTarget,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    total = 0
    stokes = []

    def process(config: int, batch: int, args) -> None:
        nonlocal total
        result = recorder.queue.view(config)
        result = result[: result.count]
        # undo attenuation
        vg = 1.0 / model.ng * u.c
        d = vg * (result["time"] - t0)
        value0 = result["contrib"] * np.exp(mu_a * d)
        # add to sum
        total += value0.sum()

        # copy stokes if polarized
        if polarized:
            stokes.append(result["stokes"].copy())

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create batches
    tasks = [
        {},
    ] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()

    # calculate direct contribution without attenuation
    directContrib = budget * np.exp(-mu_s * radius)
    # calculate expected contribution
    contrib = budget - directContrib if disableDirect else budget

    # check for energy conservation
    estimate = total / (batch_size * n_batches)
    assert np.abs(estimate / contrib - 1.0) < err

    # if polarized: check stokes vector is valid
    if polarized:
        stokes = np.concatenate(stokes)
        assert np.abs(stokes[:, 0] - 1.0).max() < 1e-5
        assert stokes[:, 1:].max() <= 1.0
        assert stokes[:, 1:].min() >= -1.0
        assert np.all(np.square(stokes[:, 1:]).sum(-1) <= 1.0)


@pytest.mark.parametrize(
    "mu_a,mu_s,mu_sample,g,disableDirect,polarized,err",
    [
        (0.0, 0.01, 0.05, 0.0, False, False, 5.5e-3),
        (0.05, 0.005, 0.05, 0.0, False, False, 3.0e-3),
        (0.05, 0.005, np.nan, 0.0, False, False, 3.0e-3),
        (0.0, 0.01, 0.05, 0.0, False, True, 5.1e-3),
        (0.05, 0.005, 0.05, 0.0, True, True, 4.8e-3),
        (0.05, 0.005, 0.05, 0.5, True, False, 8.0e-3),
        (0.01, 0.01, 0.05, -0.5, False, False, 2.3e-3),
    ],
)
def test_VolumeBackwardTracer(
    mu_a: float,
    mu_s: float,
    mu_sample: float | None,
    g: float,
    disableDirect: bool,
    polarized: bool,
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
    medium = model.createMedium()
    store = theia.material.MaterialStore([], media=[medium])

    # create scene
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=position, timeRange=(t0, t0), budget=budget
    )
    camera = theia.camera.SphereCamera(position=position, radius=-radius)
    # make target a bit larger to not occlude the camera
    target = InnerSphereTarget(position=position, radius=radius * 1.001)
    recorder = theia.response.HitRecorder(polarized=polarized)
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.VolumeBackwardTracer(
        batch_size,
        light,
        camera,
        photons,
        recorder,
        rng,
        medium=store.media["homogenous"],
        nScattering=max_length,
        callback=stats,
        target=target,
        scatterCoefficient=mu_sample,
        maxTime=maxTime,
        disableDirectLighting=disableDirect,
        polarized=polarized,
    )
    rng.autoAdvance = tracer.nRNGSamples

    # create pipeline + scheduler
    total = 0
    stokes = []

    def process(config: int, batch: int, args) -> None:
        nonlocal total
        result = recorder.queue.view(config)
        result = result[: result.count]
        # undo attenuation
        vg = 1.0 / model.ng * u.c
        d = vg * (result["time"] - t0)
        value0 = result["contrib"] * np.exp(mu_a * d)
        # add to sum
        total += value0.sum()

        # copy stokes if polarized
        if polarized:
            stokes.append(result["stokes"].copy())

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create batches
    tasks = [
        {},
    ] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()

    # calculate direct contribution without attenuation
    directContrib = budget * np.exp(-mu_s * radius)
    # calculate expected contribution
    contrib = budget - directContrib if disableDirect else budget

    # check for energy conservation
    estimate = total / (batch_size * n_batches)
    assert np.abs(estimate / contrib - 1.0) < err

    # if polarized: check stokes vector is valid
    if polarized:
        stokes = np.concatenate(stokes)
        assert np.abs(stokes[:, 0] - 1.0).max() < 1e-5
        assert stokes[:, 1:].max() <= 1.0
        assert stokes[:, 1:].min() >= -1.0
        assert np.all(np.square(stokes[:, 1:]).sum(-1) <= 1.0)


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
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = theia.scene.Scene(
        [target],
        matStore.material,
        medium=matStore.media["homogenous"],
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
    recorder = theia.response.HitRecorder(polarized=polarized)
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

    def process(config: int, batch: int, args) -> None:
        nonlocal total
        result = recorder.queue.view(config)
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

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create batches
    tasks = [
        {},
    ] * n_batches  # rng advances on its own, so we dont have to update any params
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()

    # estimate for single scatter contributions
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    recorder = theia.response.HitRecorder(polarized=polarized)
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

    def process(config: int, batch: int, args) -> None:
        nonlocal singleTotal
        result = recorder.queue.view(config)
        result = result[: result.count]
        # undo attenuation
        vg = 1.0 / model.ng * u.c
        d = vg * (result["time"] - t0)
        value0 = result["contrib"] * np.exp(mu_a * d)
        # add to sum
        singleTotal += value0.sum()

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create batches
    tasks = [
        {},
    ] * 50  # rng advances on its own, so we dont have to update any params
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()

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
    recorder = theia.response.HitRecorder(polarized=polarized)
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

    def process(config: int, batch: int, args) -> None:
        batch = dumpQueue(recorder.queue.view(config))
        batches.append(batch)

    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create batches
    tasks = [
        {},
    ] * n_batches
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()

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


@pytest.mark.parametrize(
    "mu_a,mu_s,g,polarized",
    [
        (0.0, 0.05, 0.0, False),
        (0.05, 0.05, 0.0, False),
        (0.05, 0.01, -0.9, False),
        (0.05, 0.01, 0.9, False),
        # FIXME: Polarization seems broken for now
        # (0.0, 0.05, 0.0, True),
        (0.05, 0.01, -0.9, True),
    ],
)
def test_ScenePhotonTracer(mu_a: float, mu_s: float, g: float, polarized: bool):
    if not hp.isRaytracingEnabled():
        pytest.skip("ray tracing is not supported")

    # Scene settings
    position = (12.0, 15.0, 0.2) * u.m
    radius = 60.0 * u.m
    # Light settings
    budget = 1e6
    t0 = 10.0 * u.ns
    lam = 400.0 * u.nm  # doesn't really matter
    # tracer settings
    max_length = 80
    scatter_coef = 0.05
    maxTime = float("inf")
    n_runs = 20
    n_scatterPerRun = 5
    # simulation settings
    batch_size = 1 * 1024 * 1024
    n_batches = 20

    # create materials
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium()
    material = theia.material.Material("det", medium, None, flags="DB")
    matStore = theia.material.MaterialStore([material])
    # load meshes
    meshStore = theia.scene.MeshStore({"sphere": "assets/sphere.stl"})
    # create scene
    trafo = Transform.TRS(scale=radius, translate=position)
    target = meshStore.createInstance("sphere", "det", trafo, detectorId=0)
    scene = theia.scene.Scene(
        [target],
        matStore.material,
        medium=matStore.media["homogenous"],
    )

    # create tracer
    rng = theia.random.PhiloxRNG(key=0xC01DC0FFEE)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=position, timeRange=(t0, t0), budget=budget
    )
    # use scene tracer as known truth source for estimating the amount hits
    hist_response = theia.response.HistogramHitResponse(
        theia.response.UniformValueResponse(),
        binSize=50.0 * u.ns,
        nBins=400,
    )
    hist_tracer = theia.trace.SceneForwardTracer(
        batch_size,
        light,
        photons,
        hist_response,
        rng,
        scene,
        maxPathLength=max_length,
        scatterCoefficient=scatter_coef,
        maxTime=maxTime,
        polarized=polarized,
    )
    rng.autoAdvance = hist_tracer.nRNGSamples
    # create pipeline + scheduler
    hists = []
    process = lambda c, b, _: hists.append(hist_response.result(c).copy())
    hist_pipeline = pl.Pipeline(hist_tracer.collectStages())
    scheduler = pl.PipelineScheduler(hist_pipeline, processFn=process)

    # create batches
    tasks = [{} for i in range(n_batches)]
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()

    # get total amount of expected hits
    hist = np.mean(hists, 0)
    exp_hits = hist.sum()

    # next we can do the photon tracer
    response = theia.response.StoreTimeHitResponse(
        theia.response.UniformValueResponse()
    )
    # response = theia.response.HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.ScenePhotonTracer(
        int(budget),  # this way a single batch is enough
        light,
        photons,
        response,
        rng,
        scene,
        nScatteringPerRun=n_scatterPerRun,
        nRuns=n_runs,
        maxTime=maxTime,
        polarized=polarized,
        callback=stats,
    )
    # run pipeline only once
    pl.runPipeline(tracer.collectStages())
    hits = response.result(0)
    # hits = response.queue.view(0)
    # check amount
    assert hits is not None
    if mu_a != 0.0:
        # since the photon tracer samples a random distribution we need to allow a
        # rather large error. here we use 5 sigmas as a rather pessimistic test
        assert np.abs(hits.count - exp_hits) < 5.0 * np.sqrt(exp_hits)
    else:
        # no absorption means we should recover all photons
        assert hits.count == int(budget)


@pytest.mark.parametrize(
    "mu_a,mu_s,g,polarized",
    [
        (0.0, 0.05, 0.0, False),
        (0.05, 0.05, 0.0, False),
        (0.05, 0.01, -0.9, False),
        (0.05, 0.01, 0.9, False),
        # FIXME: Polarization seems broken for now
        # (0.0, 0.05, 0.0, True),
        (0.05, 0.01, -0.9, True),
    ],
)
def test_VolumePhotonTracer(mu_a: float, mu_s: float, g: float, polarized: bool):
    # Scene settings
    position = (12.0, 15.0, 0.2) * u.m
    radius = 60.0 * u.m
    # light settings
    budget = 1e6
    t0 = 10.0 * u.ns
    lam = 400.0 * u.nm
    # tracer settings
    max_length = 80
    maxTime = float("inf")
    scatter_coef = 0.05
    n_runs = 40
    n_scatterPerRun = 5
    # simulation settings
    # batch_size = 2 * 1024 * 1024 # does not work on my laptop!?
    batch_size = 1 * 1024 * 1024
    n_batches = 20

    # create medium
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium()
    store = theia.material.MaterialStore([], media=[medium])

    # create scene
    rng = theia.random.PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.UniformWavelengthSource(lambdaRange=(lam, lam))
    light = theia.light.SphericalLightSource(
        position=position, timeRange=(t0, t0), budget=budget
    )
    target = InnerSphereTarget(position=position, radius=radius)
    # use volume tracer as known truth source for estimating the amount of hits
    hist_response = theia.response.HistogramHitResponse(
        theia.response.UniformValueResponse(),
        binSize=50.0 * u.ns,
        nBins=400,
    )
    hist_tracer = theia.trace.VolumeForwardTracer(
        batch_size,
        light,
        target,
        photons,
        hist_response,
        rng,
        medium=store.media["homogenous"],
        maxTime=maxTime,
        nScattering=max_length,
        scatterCoefficient=scatter_coef,
        polarized=polarized,
    )
    rng.autoAdvance = hist_tracer.nRNGSamples
    # create pipeline + scheduler
    hists = []
    process = lambda c, b, _: hists.append(hist_response.result(c).copy())
    hist_pipeline = pl.Pipeline(hist_tracer.collectStages())
    scheduler = pl.PipelineScheduler(hist_pipeline, processFn=process)

    # create batches
    tasks = [{} for i in range(n_batches)]
    scheduler.schedule(tasks)
    scheduler.wait()
    # destroy scheduler to allow for freeing resources
    scheduler.destroy()

    # get total amount of expected hits
    hist = np.mean(hists, 0)
    exp_hits = hist.sum()

    # next we can do the photon tracer
    response = theia.response.StoreTimeHitResponse(
        theia.response.UniformValueResponse()
    )
    # response = theia.response.HitRecorder()
    stats = theia.trace.EventStatisticCallback()
    tracer = theia.trace.VolumePhotonTracer(
        int(budget),  # this way a single batch is enough
        light,
        target,
        photons,
        response,
        rng,
        medium=store.media["homogenous"],
        nScatteringPerRun=n_scatterPerRun,
        nRuns=n_runs,
        maxTime=maxTime,
        polarized=polarized,
        callback=stats,
    )
    # run pipeline only once
    pl.runPipeline(tracer.collectStages())
    hits = response.result(0)
    # hits = response.queue.view(0)
    # check amount
    assert hits is not None
    if mu_a != 0.0:
        # since the photon tracer samples a random distribution we need to allow a
        # rather large error. here we use 5 sigmas as a rather pessimistic test
        assert np.abs(hits.count - exp_hits) < 5.0 * np.sqrt(exp_hits)
    else:
        # no absorption means we should recover all photons
        assert hits.count == int(budget)
