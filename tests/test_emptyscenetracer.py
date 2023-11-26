import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl

import theia.items
import theia.light
import theia.material
import theia.scene
import theia.trace

from hephaistos.queue import as_queue
from theia.random import PhiloxRNG

from common.models import WaterModel


def test_emptyscenetracer_unbufferd():
    N = 32 * 256
    N_PHOTONS = 4
    light_pos = (-1.0, -7.0, 0.0)
    light_intensity = 10.0
    target_pos, target_radius = (5.0, 2.0, -8.0), 4.0
    target = theia.scene.SphereBBox(target_pos, target_radius)

    # create water medium
    water = WaterModel().createMedium()
    tensor, _, media = theia.material.bakeMaterials(media=[water])

    # create pipeline
    philox = PhiloxRNG(key=0xC0FFEEC0FFEE)
    tracer = theia.trace.EmptySceneTracer(
        N, rng=philox, target=target, nPhotons=N_PHOTONS
    )
    light = theia.light.SphericalLightSource(
        N,
        medium=media["water"],
        rng=philox,
        rayQueue=tracer.rayQueueIn,
        position=light_pos,
        intensity=light_intensity,
        nPhotons=N_PHOTONS,
    )
    retriever = pl.RetrieveTensorStage(tracer.hitQueue)
    pipeline = pl.Pipeline([philox, light, tracer, retriever])
    # run pipeline
    pipeline.run(0)

    # retrieve hits
    hitItem = theia.items.createHitQueueItem(N_PHOTONS)
    hits = as_queue(retriever.buffer(0), hitItem)
    assert hits.count > 0  # catch degenerate case
    hits = hits[: hits.count]

    # check result
    assert np.allclose(np.square(hits["position"]).sum(-1), target_radius**2)
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    # TODO: more sophisticated tests


def test_emptyscenetracer_buffered():
    N = 32 * 256
    N_PHOTONS = 4
    light_pos = (-1.0, -7.0, 0.0)
    light_intensity = 10.0
    target_pos, target_radius = (5.0, 2.0, -8.0), 4.0
    target = theia.scene.SphereBBox(target_pos, target_radius)

    # create water medium
    water = WaterModel().createMedium()
    tensor, _, media = theia.material.bakeMaterials(media=[water])

    # create pipeline
    philox = PhiloxRNG(key=0xC0FFEEC0FFEE)
    tracer = theia.trace.EmptySceneTracer(
        N,
        rng=philox,
        target=target,
        nPhotons=N_PHOTONS,
        nScattering=2,
        nIterations=3,
        keepRays=True,
    )
    light = theia.light.SphericalLightSource(
        N,
        medium=media["water"],
        rng=philox,
        rayQueue=tracer.rayQueueIn,
        position=light_pos,
        intensity=light_intensity,
        nPhotons=N_PHOTONS,
    )
    fetchHits = pl.RetrieveTensorStage(tracer.hitQueue)
    fetchRays = pl.RetrieveTensorStage(tracer.rayQueueOut)
    pipeline = pl.Pipeline([philox, light, tracer, fetchHits, fetchRays])
    # run pipeline
    pipeline.run(0)

    # retrieve hits
    hitItem = theia.items.createHitQueueItem(N_PHOTONS)
    hits = as_queue(fetchHits.buffer(0), hitItem)
    # retrieve rays
    rayItem = theia.items.createRayQueueItem(N_PHOTONS)
    rays = as_queue(fetchRays.buffer(0), rayItem)

    # check result
    assert hits.count == N * tracer.nIterations * tracer.nScattering
    hits = hits[: hits.count]
    assert np.allclose(np.square(hits["position"]).sum(-1), target_radius**2)
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    # TODO: more sophisticated tests

    # check result
    assert rays.count > 0
    rays = rays[: rays.count]
    lamMin, lamMax = light.getParam("lambdaRange")
    assert np.all((rays["time"] > 0.0) & (rays["time"] < tracer.getParam("maxTime")))
    assert np.all((rays["wavelength"] >= lamMin) & (rays["wavelength"] <= lamMax))
    # TODO: more sophisticated tests
