import pytest

import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl
from hephaistos.queue import dumpQueue

import theia
import theia.estimator
import theia.light
import theia.material
import theia.random
import theia.scene
import theia.trace
import theia.units as u

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
        (0.0, 0.005, 0.0, True),
        (0.05, 0.01, -0.9, True),
    ],
)
def test_SceneTracer_EnergyConserved(
    mu_a: float, mu_s: float, g: float, polarized: bool
) -> None:
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
    n_batches = 100

    # create materials
    model = MediumModel(mu_a, mu_s, g)
    medium = model.createMedium()
    material = theia.material.Material("det", medium, None, flags="DB")
    matStore = theia.material.MaterialStore([material])
    # load meshes
    meshStore = theia.scene.MeshStore({"sphere": "assets/sphere.stl"})

    # create scene
    targets = [theia.scene.SphereBBox(position, radius)]
    trafo = theia.scene.Transform.Scale(radius, radius, radius).translate(*position)
    target = meshStore.createInstance("sphere", "det", transform=trafo, detectorId=0)
    scene = theia.scene.Scene(
        [target],
        matStore.material,
        medium=matStore.media["homogenous"],
        targets=targets,
    )
    # create light (delta pulse)
    photons = theia.light.UniformPhotonSource(
        lambdaRange=(lam, lam), timeRange=(t0, t0), budget=budget
    )
    light = theia.light.SphericalLightSource(photons, position=position)
    # create tracer
    rng = theia.random.SobolQRNG(seed=0xC0FFEE)
    recorder = theia.estimator.HitRecorder(polarized=polarized)
    tracer = theia.trace.SceneTracer(
        batch_size,
        light,
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
    assert (
        estimate < budget
    )  # we are biased by ignoring longer paths, i.e. loose a bit of energy
    assert np.abs(estimate / budget - 1.0) < 0.05

    # additional check since we have the data: uniform hits on sphere
    positions = np.concatenate([b["position"] for b in batches], axis=0)
    assert np.abs(positions.mean(0)).max() < 5e-3
    assert np.abs(positions.var(0) - 1 / 3).max() < 0.1

    # if polarized: check stokes vector is valid
    if polarized:
        stokes = np.concatenate([b["stokes"] for b in batches])
        assert np.abs(stokes[:, 0] - 1.0).max() < 1e-5
        assert stokes[:, 1:].max() <= 1.0
        assert stokes[:, 1:].min() >= -1.0
        assert np.all(np.square(stokes[:, 1:]).sum(-1) <= 1.0)
