import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl

import theia.estimator
import theia.items
import theia.light
import theia.material
import theia.scene
import theia.trace

from hephaistos.queue import as_queue
from theia.random import PhiloxRNG

from common.models import WaterModel


def test_scenetracer_unbufferd():
    N = 32 * 256
    N_PHOTONS = 4
    light_pos = (-1.0, -7.0, 0.0)
    light_intensity = 1000.0

    # create materials
    water = WaterModel().createMedium()
    glass = theia.material.BK7Model().createMedium()
    mat = theia.material.Material(
        "mat", glass, water, flags=theia.material.Material.TARGET_BIT
    )
    tensor, material, media = theia.material.bakeMaterials(mat)
    # create scene
    store = theia.scene.MeshStore(
        {"cube": "assets/cone.stl", "sphere": "assets/sphere.stl"}
    )
    r, d = 50.0, 3.0
    r_scale = 0.99547149974733 # radius of inscribed sphere (icosphere)
    r_insc = r * r_scale  
    x, y, z = 10.0, 5.0, 0.0
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

    # create pipeline
    philox = PhiloxRNG(key=0xC0FFEEC0FFEE)
    tracer = theia.trace.SceneTracer(
        N, scene, rng=philox, nPhotons=N_PHOTONS, targetIdx=1
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
    r_hits = np.sqrt(np.square(hits["position"]).sum(-1))
    assert np.all((r_hits <= 1.0) & (r_hits >= r_scale))
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    # TODO: more sophisticated tests

def test_scenetracer_buffered():
    N = 32 * 256
    N_PHOTONS = 4
    light_pos = (-1.0, -7.0, 0.0)
    light_intensity = 1000.0

    # create materials
    water = WaterModel().createMedium()
    glass = theia.material.BK7Model().createMedium()
    mat = theia.material.Material(
        "mat", glass, water, flags=theia.material.Material.TARGET_BIT
    )
    tensor, material, media = theia.material.bakeMaterials(mat)
    # create scene
    store = theia.scene.MeshStore(
        {"cube": "assets/cone.stl", "sphere": "assets/sphere.stl"}
    )
    r, d = 50.0, 3.0
    r_scale = 0.99547149974733 # radius of inscribed sphere (icosphere)
    r_insc = r * r_scale  
    x, y, z = 10.0, 5.0, 0.0
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

    # create pipeline
    philox = PhiloxRNG(key=0xC0FFEEC0FFEE)
    tracer = theia.trace.SceneTracer(
        N, scene,
        rng=philox,
        targetIdx=1,
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
    r_hits = np.sqrt(np.square(hits["position"]).sum(-1))
    assert np.all((r_hits <= 1.0) & (r_hits >= r_scale))
    assert np.allclose(np.square(hits["normal"]).sum(-1), 1.0)
    # TODO: more sophisticated tests

    # check result
    assert rays.count > 0
    rays = rays[: rays.count]
    lamMin, lamMax = light.getParam("lambdaRange")
    assert np.all((rays["time"] > 0.0) & (rays["time"] < tracer.getParam("maxTime")))
    assert np.all((rays["wavelength"] >= lamMin) & (rays["wavelength"] <= lamMax))
    # TODO: more sophisticated tests


def test_scenetracer_pipeline():
    N = 32 * 256
    N_BINS = 128
    N_PHOTONS = 4
    BIN_SIZE = 2.0
    T0 = 0.0
    T1 = T0 + N_BINS * BIN_SIZE
    light_pos = (-1.0, -7.0, 0.0)
    light_intensity = 1000.0

    # create materials
    water = WaterModel().createMedium()
    glass = theia.material.BK7Model().createMedium()
    mat = theia.material.Material(
        "mat", glass, water, flags=theia.material.Material.TARGET_BIT
    )
    tensor, material, media = theia.material.bakeMaterials(mat)
    # create scene
    store = theia.scene.MeshStore(
        {"cube": "assets/cone.stl", "sphere": "assets/sphere.stl"}
    )
    r, d = 50.0, 3.0
    r_scale = 0.99547149974733 # radius of inscribed sphere (icosphere)
    r_insc = r * r_scale  
    x, y, z = 10.0, 5.0, 0.0
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

    # create pipeline
    philox = PhiloxRNG(key=0xC0FFEEC0FFEE)
    tracer = theia.trace.SceneTracer(
        N, scene,
        rng=philox,
        targetIdx=1,
        nPhotons=N_PHOTONS,
        nScattering=2,
        nIterations=3,
        keepRays=True,
        scatterCoefficient=0.01,
    )
    light = theia.light.SphericalLightSource(
        N,
        nPhotons=N_PHOTONS,
        medium=media["water"],
        rng=philox,
        rayQueue=tracer.rayQueueIn,
        position=light_pos,
        intensity=light_intensity,
    )
    estimator = theia.estimator.HistogramEstimator(
        tracer.maxSamples,
        tracer.hitQueue,
        detectorId=0,
        t0=T0,
        nBins=N_BINS,
        binSize=BIN_SIZE,
        clearQueue=False,
        normalization=1.0 / N,
    )
    fetch_hits = pl.RetrieveTensorStage(tracer.hitQueue)
    fetch_hist = pl.RetrieveTensorStage(estimator.histogram)
    # assemble pipeline
    pipeline = pl.Pipeline([philox, light, tracer, fetch_hits, estimator, fetch_hist])
    # run pipeline once
    pipeline.run(0)

    # fetch results
    hitItem = theia.items.createHitQueueItem(N_PHOTONS)
    hits = as_queue(fetch_hits.buffer(0), hitItem)
    hist = fetch_hist.view(0)
    # check for degenerate case
    assert hits.count > 0

    # calculated expected result
    cosine = -np.multiply(hits["normal"], hits["direction"]).sum(-1)
    weight = (hits["contribution"] * cosine[:, None]).flatten()
    t = hits["time"].flatten()
    hist_exp, _ = np.histogram(t, N_BINS, (T0, T1), weights=weight)
    hist_exp = hist_exp * estimator.norm
    # check result
    assert np.allclose(hist, hist_exp)    
