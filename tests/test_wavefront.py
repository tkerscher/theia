import numpy as np
import hephaistos as hp

from hephaistos.glsl import vec3, stackVector
from hephaistos.queue import QueueBuffer, QueueTensor
from hephaistos.pipeline import runPipelineStage

import theia.items
import theia.light
import theia.material
import theia.scene
import theia.trace

from theia.random import PhiloxRNG
from theia.util import compileShader, packUint64

from common.models import WaterModel
from ctypes import *

from numpy.lib.recfunctions import structured_to_unstructured

N = 32 * 256
N_PHOTONS = 4

preamble = f"""
#define LOCAL_SIZE 32
#define N_PHOTONS {N_PHOTONS}
#define QUEUE_SIZE {N}
#define HIT_QUEUE_SIZE {N}
"""


class Detector(Structure):
    _fields_ = [("position", vec3), ("radius", c_float)]


class TraceParams(Structure):
    _fields_ = [
        ("targetIdx", c_uint32),
        ("scatterCoefficient", c_float),
        ("maxTime", c_float),
        ("lowerBBoxCorner", vec3),
        ("upperBBoxCorner", vec3),
    ]


def test_wavefront_trace(rng):
    N = 32 * 256
    # create rng
    philox = PhiloxRNG(key=0xC01DC0FFEE)  # need 2N samples
    philox.update(0)  # update manually
    # create programs
    headers = {"rng.glsl": philox.sourceCode}
    wavefront_trace = hp.Program(
        compileShader("wavefront.trace.glsl", preamble, headers)
    )
    wavefront_intersect = hp.Program(
        compileShader("wavefront.intersection.glsl", preamble, headers=headers)
    )

    # create material
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
    r_insc = r * 0.99547149974733  # radius of inscribed sphere (icosphere)
    x, y, z = 10.0, 5.0, 0.0
    t1 = theia.scene.Transform.Scale(r, r, r).translate(x, y, z + r + d)
    c1 = store.createInstance("sphere", "mat", transform=t1, detectorId=0)
    t2 = theia.scene.Transform.Scale(r, r, r).translate(x, y, z - r - d)
    c2 = store.createInstance("sphere", "mat", transform=t2, detectorId=1)
    scene = theia.scene.Scene([c1, c2], material, medium=media["water"])

    # create light
    rayItem = theia.items.createRayQueueItem(N_PHOTONS)
    rayTensor = QueueTensor(rayItem, N)
    light = theia.light.HostLightSource(
        N, rayQueue=rayTensor, nPhotons=4, medium=media["water"]
    )
    # fill input buffer
    queries = light.view(0)
    theta = rng.random(N) * np.pi
    phi = rng.random(N) * 2.0 * np.pi
    queries["direction"] = np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
        axis=-1,
    )
    pos = np.array([10.0, 5.0, 0.0])
    queries["position"][:, :] = pos
    queries["wavelength"] = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    # Use first wavelength as constant for reordering results
    queries["wavelength"][:, 0] = np.linspace(200.0, 800.0, N)
    queries["log_contrib"] = rng.random((N, N_PHOTONS)) * 5.0 - 4.0
    queries["startTime"] = rng.random((N, N_PHOTONS)) * 50.0
    queries["lin_contrib"] = rng.random((N, N_PHOTONS))
    # fill ray queue
    runPipelineStage(light)

    # reserve memory for
    rayItem = theia.items.createRayQueueItem(N_PHOTONS)
    rayBufferIn = QueueBuffer(rayItem, N)
    rayBufferOut = QueueBuffer(rayItem, N)
    intItem = theia.items.createIntersectionQueueItem(N_PHOTONS)
    intBuffer = QueueBuffer(intItem, N)
    intTensor = QueueTensor(intItem, N)
    volItem = theia.items.createVolumeScatterQueueItem(N_PHOTONS)
    volBuffer = QueueBuffer(volItem, N)
    volTensor = QueueTensor(volItem, N)
    hitItem = theia.items.createHitQueueItem(N_PHOTONS)
    hitBuffer = QueueBuffer(hitItem, N)
    hitTensor = QueueTensor(hitItem, N)
    paramsBuffer = hp.StructureBuffer(TraceParams)
    paramsTensor = hp.StructureTensor(TraceParams)
    # bind memory
    light.rayQueue = rayTensor
    philox.bindParams(wavefront_trace, 0)
    wavefront_trace.bindParams(
        RayQueueBuffer=rayTensor,
        IntersectionQueueBuffer=intTensor,
        VolumeScatterQueueBuffer=volTensor,
        tlas=scene.tlas,
        Params=paramsTensor,
    )
    wavefront_intersect.bindParams(
        IntersectionQueueBuffer=intTensor,
        RayQueueBuffer=rayTensor,
        HitQueueBuffer=hitTensor,
        Geometries=scene.geometries,
        Params=paramsTensor,
    )

    # create params
    lambda_sc = 1.0 / 100.0
    t_max = 500.0
    paramsBuffer.targetIdx = 1
    paramsBuffer.scatterCoefficient = lambda_sc
    paramsBuffer.maxTime = t_max
    paramsBuffer.lowerBBoxCorner = vec3(x=-100.0, y=-100.0, z=-100.0)
    paramsBuffer.upperBBoxCorner = vec3(x=100.0, y=100.0, z=100.0)

    # run program
    (
        hp.beginSequence()
        .And(hp.updateTensor(paramsBuffer, paramsTensor))
        .And(hp.clearTensor(intTensor))
        .And(hp.clearTensor(volTensor))
        .And(hp.retrieveTensor(rayTensor, rayBufferIn))
        .Then(wavefront_trace.dispatch(N // 32))
        .Then(hp.retrieveTensor(intTensor, intBuffer))
        .And(hp.retrieveTensor(volTensor, volBuffer))
        .And(hp.clearTensor(rayTensor))
        .Then(wavefront_intersect.dispatch(N // 32))
        .Then(hp.retrieveTensor(hitTensor, hitBuffer))
        .And(hp.retrieveTensor(rayTensor, rayBufferOut))
        .Submit()
        .wait()
    )

    # check result
    r_in = rayBufferIn.view
    intQueue = intBuffer.view
    volQueue = volBuffer.view
    rayQueue = rayBufferOut.view
    assert intQueue.count + volQueue.count == N  # all items processed?
    assert rayQueue.count <= intQueue.count

    # check volume items did not pass through geometry
    v = volQueue[: volQueue.count]
    cos_theta = np.abs(v["direction"][:, 2])  # dot with +/- e_z
    cos_theta_min = np.sqrt(1.0 - (r_insc / (r_insc + d)) ** 2)  # tangent angle
    may_hit = cos_theta >= cos_theta_min + 0.01  # acount for err in geometry too
    cos_theta = cos_theta[may_hit]
    t0 = (r_insc + d) * cos_theta - np.sqrt(
        (r_insc + d) ** 2 * cos_theta**2 - (2 * r_insc * d + d**2)
    )
    # bit tricky to test, since our sphere is approximated with triangles
    assert (v["dist"][may_hit] < t0).sum() / may_hit.sum() > 0.95
    # limit outliers
    assert ((v["dist"][may_hit] - t0) / t0).max() < 0.1

    # check intersections
    i = rayQueue[: rayQueue.count]
    # reconstruct input queries
    i_idx = (i["wavelength"][:, 0] - 200.0) / (600.0 / (N - 1))
    i_in = r_in[np.round(i_idx).astype(np.int32)]
    # check position: direction match
    i_dir = i["position"] - i_in["position"]
    i_dir /= np.sqrt(np.square(i_dir).sum(-1))[:, None]
    assert np.abs(i_dir - i_in["direction"]).max() < 1e-4
    # check positions: on sphere
    center = np.array([x, y, r + d])
    i_dup = np.square(i["position"] - center[None, :]).sum(-1)
    center[2] = -(r + d)
    i_d = np.square(i["position"] - center[None, :]).sum(-1)
    mask = i_in["direction"][:, 2] > 0.0
    i_d[mask] = i_dup[mask]
    i_d = np.sqrt(i_d)
    # on small sphere
    d_small = (i_d <= r + 5e-5) & (i_d >= r_insc + 5e-5)
    d_big = (i_d <= 2 * r + 5e-5) & (i_d >= 2 * r_insc + 5e-5)
    assert np.all(d_small | d_big)
    # check distance & throughput
    i_dist = np.sqrt(np.square(i["position"] - pos[None, :]).sum(-1))
    i_t = i_dist[:, None] / i_in["vg"] + i_in["time"]
    assert np.allclose(i_t, i["time"])
    i_thr = (lambda_sc - i_in["mu_e"]) * i_dist[:, None]
    i_thr += i_in["log_contrib"]
    assert np.allclose(i_thr, i["log_contrib"], rtol=5e-5)
    # TODO: check reflectance
    # check direction
    i_normal = (i["position"] - center[None, :]) / i_d[:, None]
    center[2] = r + d
    i_normal_up = (i["position"] - center[None, :]) / i_d[:, None]
    i_normal[mask] = i_normal_up[mask]
    i_dout = (
        i_in["direction"]
        - 2.0 * np.multiply(i_in["direction"], i_normal).sum(-1)[:, None] * i_normal
    )  # reflect
    n_err = np.abs(i_dout - i["direction"])
    assert n_err.max() < 0.01  # TODO: error a bit hight

    # check detector hits
    hitQueue = hitBuffer.view
    hits = hitQueue[: hitQueue.count]
    # reconstruct input queries
    hits_idx = (hits["wavelength"][:, 0] - 200.0) / (600.0 / (N - 1))
    hits_in = r_in[np.round(hits_idx).astype(np.int32)]
    # check positions (note: unit icosphere local coordinates)
    hits_d = np.sqrt(np.square(hits["position"]).sum(-1))
    assert np.all((hits_d <= 1.0 + 5e-5) & (hits_d >= 0.99547149974733 + 5e-5))
    # TODO: check distance & throughput
    # TODO: Calc linear throughput and final throughput
    # check normal (note: unit icosphere)
    hits_normal = hits["position"] / hits_d[:, None]
    n_err = np.abs(hits_normal - hits["normal"])
    assert n_err.max() < 0.01
    # test direction
    n_err = np.abs(hits_in["direction"] - hits["direction"])
    assert n_err.max() < 1e-5
    # check the hits are on the lower sphere via ray direction
    assert hits_in["direction"][:, 2].max() < 0.0


def test_wavefront_volume(rng):
    N = 32 * 256
    N_DET = 32
    # create rng
    philox = PhiloxRNG(key=0xC01DC0FFEE)  # need 2N samples
    philox.update(0)  # update manually
    # create programs
    headers = {"rng.glsl": philox.sourceCode}
    program = hp.Program(compileShader("wavefront.volume.glsl", preamble, headers))

    # create medium
    model = WaterModel()
    water = model.createMedium()
    tensor, _, media = theia.material.bakeMaterials(media=[water])

    # allocate memory
    volItem = theia.items.createVolumeScatterQueueItem(N_PHOTONS)
    volumeQueueBuffer = QueueBuffer(volItem, N)
    volumeQueueTensor = QueueTensor(volItem, N)
    rayItem = theia.items.createRayQueueItem(N_PHOTONS)
    rayQueueBuffer = QueueBuffer(rayItem, N)
    rayQueueTensor = QueueTensor(rayItem, N)
    shadowItem = theia.items.createShadowQueueItem(N_PHOTONS)
    shadowQueueBuffer = QueueBuffer(shadowItem, N)
    shadowQueueTensor = QueueTensor(shadowItem, N)
    detectorBuffer = hp.ArrayBuffer(Detector, N_DET)
    detectorTensor = hp.ArrayTensor(Detector, N_DET)
    paramsBuffer = hp.StructureBuffer(TraceParams)
    paramsTensor = hp.StructureTensor(TraceParams)
    # bind memory
    philox.bindParams(program, 0)
    program.bindParams(
        VolumeScatterQueueBuffer=volumeQueueTensor,
        RayQueueBuffer=rayQueueTensor,
        ShadowQueueBuffer=shadowQueueTensor,
        Detectors=detectorTensor,
        Params=paramsTensor,
    )

    # set params
    t_max = 80.0
    _lam = 10.0
    mu_s = 0.6
    mu_e = 0.9
    targetIdx = 13
    paramsBuffer.targetIdx = targetIdx
    paramsBuffer.scatterCoefficient = _lam
    paramsBuffer.maxTime = t_max
    paramsBuffer.lowerBBoxCorner = vec3(x=-50.0, y=-50.0, z=-50.0)
    paramsBuffer.upperBBoxCorner = vec3(x=50.0, y=50.0, z=50.0)
    # create detectors
    det = detectorBuffer.numpy()
    det["position"] = stackVector(
        [
            rng.random(N_DET) * 20.0 - 10.0,
            rng.random(N_DET) * 20.0 - 10.0,
            rng.random(N_DET) * 20.0 - 10.0,
        ],
        vec3,
    )
    det["radius"] = rng.random(N_DET) * 4.0 + 1.0
    # fill input
    N_in = N - 146
    v = volumeQueueBuffer.view
    v.count = N_in
    x = rng.random(N) * 10.0 - 5.0
    y = rng.random(N) * 10.0 - 5.0
    z = rng.random(N) * 10.0 - 5.0
    phi = rng.random(N) * 2.0 * np.pi
    cos_theta = 2.0 * rng.random(N) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    v["position"] = np.stack([x, y, z], -1)
    v["direction"] = np.stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta), axis=-1
    )
    v["rngStream"] = np.arange(N)
    v["rngCount"] = rng.integers(0, 12, N)
    medium = packUint64(media["water"])
    v["medium"] = (medium.x, medium.y)
    v["wavelength"] = rng.random((N, N_PHOTONS)) * 500.0 + 200.0
    # Use first wavelength as constant for reordering results
    v["wavelength"][:, 0] = np.linspace(200.0, 800.0, N)
    v["time"] = rng.random((N, N_PHOTONS)) * 20.0
    v["lin_contrib"] = rng.random((N, N_PHOTONS))
    v["log_contrib"] = rng.random((N, N_PHOTONS)) * 5.0 - 3.0
    v["vg"] = rng.random((N, N_PHOTONS)) * 0.5 + 0.5
    v["mu_s"] = mu_s
    v["mu_e"] = mu_e
    v["dist"] = rng.random(N) * 100.0

    # run program
    (
        hp.beginSequence()
        .And(hp.updateTensor(volumeQueueBuffer, volumeQueueTensor))
        .And(hp.updateTensor(paramsBuffer, paramsTensor))
        .And(hp.updateTensor(detectorBuffer, detectorTensor))
        .And(hp.clearTensor(rayQueueTensor))
        .And(hp.clearTensor(shadowQueueTensor))
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(rayQueueTensor, rayQueueBuffer))
        .And(hp.retrieveTensor(shadowQueueTensor, shadowQueueBuffer))
        .Submit()
        .wait()
    )

    # check result
    v = v[:N_in]  # throw away inputs that should not have been processed
    travelTime = v["time"] + v["dist"][:, None] / v["vg"]
    pos = v["position"] + v["dist"][:, None] * v["direction"]
    p_mask = np.all((pos >= -50.0) & (pos <= 50.0), -1)
    t_mask = np.any(travelTime <= t_max, -1)
    mask = p_mask & t_mask
    assert rayQueueBuffer.count == mask.sum()
    assert shadowQueueBuffer.count == mask.sum()
    N_res = mask.sum()

    # reconstruct item ordering
    r = rayQueueBuffer.view[:N_res]
    r_idx = (r["wavelength"][:, 0] - 200.0) / (600.0 / (N - 1))
    r = r[np.argsort(np.round(r_idx).astype(np.int32))]
    assert len(r) == N_res
    s = shadowQueueBuffer.view[:N_res]
    s_idx = (s["wavelength"][:, 0] - 200.0) / (600.0 / (N - 1))
    s = s[np.argsort(np.round(s_idx).astype(np.int32))]
    assert len(s) == N_res
    # mask away inputs that should have been terminated
    v = v[mask]
    travelTime = travelTime[mask]
    pos = pos[mask]
    # check we matched correctly
    assert np.all(r["wavelength"][:, 0] == v["wavelength"][:, 0])
    assert np.all(s["wavelength"][:, 0] == v["wavelength"][:, 0])

    # check created shadow items
    d_pos = structured_to_unstructured(det[targetIdx]["position"])
    sd_dir = d_pos - s["position"]
    center_dist = np.sqrt(np.square(sd_dir).sum(-1))
    sd_dir = sd_dir / center_dist[:, None]
    assert np.all(s["dist"] >= center_dist)
    edge_dist = np.sqrt(det[targetIdx]["radius"] ** 2 + center_dist**2)
    cos_cone = center_dist / edge_dist
    cos_theta = np.multiply(sd_dir, s["direction"]).sum(-1)
    assert np.all(cos_theta >= cos_cone)
    assert np.abs(s["position"] - pos).max() < 5e-6
    delta_rng = s["rngCount"] - v["rngCount"]
    assert delta_rng.max() == delta_rng.min()  # we have to always draw the same amount
    assert delta_rng.min() > 0  # check that we actually drew some random numbers
    assert np.abs(s["time"] - travelTime).max() < 5e-5
    s_logc = (_lam - mu_e) * v["dist"][:, None] + v["log_contrib"]
    assert np.abs(s["log_contrib"] - s_logc).max() < 1e-4  # Why this large error?
    assert np.abs(s["wavelength"] - v["wavelength"]).max() < 1e-6

    # Check created ray queries
    assert np.abs(r["position"] - pos).max() < 5e-6
    delta_rng = r["rngCount"] - v["rngCount"]
    assert delta_rng.max() == delta_rng.min()  # we have to always draw the same amount
    assert delta_rng.min() > 0  # check that we actually drew some random numbers
    assert np.abs(r["time"] - travelTime).max() < 5e-5
    # assert np.abs(r_ph["T_lin"] - mu_s / _lam).max() < 1e-6
    r_logc = (_lam - mu_e) * v["dist"][:, None] + v["log_contrib"]
    assert np.abs(r["log_contrib"] - r_logc).max() < 1e-4  # Why this large error?
    assert np.abs(r["wavelength"] - v["wavelength"]).max() < 1e-6

    # TODO: check lin_contrib (requires quite some calculations...)

    # we take a shortcut on testing the correct sampling:
    # the Henyey-Greenstein's g parameter is the expected value of 2pi*cos(theta)
    # since we IS the cosine, the mean of the cosines should be g
    # (HG is normalized to 1/2pi)
    cos_theta = np.multiply(v["direction"], r["direction"]).sum(-1)
    g_mc = cos_theta.mean()
    # give a bit more slack since we dont have that many samples
    assert np.abs(g_mc - model.g) < 6e-3

    # check if scattered phi is truly random
    # create a local euclidean cosy from input direction as one axis
    # the ratio of the projection onto the other two axis is the tan
    # since we expect uniform, we are free to rotate around the input direction
    # See prbt v4: chapter 3.3.3
    s = np.copysign(1.0, v["direction"][:, 2])
    a = -1.0 / (s + v["direction"][:, 2])
    b = a * v["direction"][:, 0] * v["direction"][:, 1]
    x_ax = np.stack(
        [1.0 + s * v["direction"][:, 0] ** 2 * a, s * b, -s * v["direction"][:, 0]], -1
    )
    y_ax = np.stack([b, s + v["direction"][:, 1] ** 2 * a, -v["direction"][:, 1]], -1)
    # normalize to be safe
    x_ax = x_ax / np.sqrt(np.square(x_ax).sum(-1))[:, None]
    y_ax = y_ax / np.sqrt(np.square(y_ax).sum(-1))[:, None]
    # project onto axis
    x = np.multiply(x_ax, r["direction"]).sum(-1)
    y = np.multiply(y_ax, r["direction"]).sum(-1)
    phi = np.arctan2(y, x)
    # check if phi is uniform
    hist = np.histogram(phi, bins=10, range=(-np.pi, np.pi))[0]
    # check for uniform
    assert (
        np.abs((hist / N_res) - (1.0 / 10)).max() < 0.01
    )  # we dont have much statistics


def test_wavefront_shadow(rng):
    N = 32 * 256

    # create program
    program = hp.Program(compileShader("wavefront.shadow.glsl", preamble))

    # create material
    water = WaterModel().createMedium()
    glass = theia.material.BK7Model().createMedium()
    flags = theia.material.Material.TARGET_BIT
    mat = theia.material.Material("mat", glass, water, flags=flags)
    tensor, material, media = theia.material.bakeMaterials(mat)

    # create scene
    store = theia.scene.MeshStore(
        {"cube": "assets/cube.stl", "sphere": "assets/sphere.stl"}
    )
    r_insc = 0.99547149974733  # diameter of inscribed sphere (icosphere)
    t1 = theia.scene.Transform.Rotation(0.0, 1.0, 0.0, 0.5 * np.pi).translate(
        2.0, 0.0, 5.0
    )
    c1 = store.createInstance("sphere", "mat", transform=t1, detectorId=1)
    t2 = theia.scene.Transform.Translation(3.0, 0.0, 2.0)
    c2 = store.createInstance("cube", "mat", transform=t2)  # obstacle
    t3 = theia.scene.Transform.Translation(5.0, 0.0, 8.0)
    c3 = store.createInstance("cube", "mat", transform=t3, detectorId=3)
    scene = theia.scene.Scene([c1, c2, c3], material, medium=media["water"])

    # reserve memory
    shadowItem = theia.items.createShadowQueueItem(N_PHOTONS)
    shadowBuffer = QueueBuffer(shadowItem, N)
    shadowTensor = QueueTensor(shadowItem, N)
    hitItem = theia.items.createHitQueueItem(N_PHOTONS)
    hitBuffer = QueueBuffer(hitItem, N)
    hitTensor = QueueTensor(hitItem, N)
    paramsBuffer = hp.StructureBuffer(TraceParams)
    paramsTensor = hp.StructureTensor(TraceParams)
    # bind params
    program.bindParams(
        ShadowQueueBuffer=shadowTensor,
        HitQueueBuffer=hitTensor,
        Geometries=scene.geometries,
        tlas=scene.tlas,
        Params=paramsTensor,
    )

    # fill input buffer
    queries = shadowBuffer.view
    queries["dist"] = 50.0
    queries["wavelength"] = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    # Use first wavelength as constant for reordering results
    queries["wavelength"][:, 0] = np.linspace(200.0, 800.0, N)
    queries["vg"] = rng.random((N, N_PHOTONS)) * 0.5 + 0.1
    queries["mu_e"] = rng.random((N, N_PHOTONS)) * 0.128
    queries["time"] = rng.random((N, N_PHOTONS)) * 50.0
    queries["lin_contrib"] = rng.random((N, N_PHOTONS))
    # all point in z direction
    queries["direction"][:, 2] = 1.0
    # swoop in x direction
    queries["position"][:, 0] = np.linspace(0.0, 7.0, N)
    # create params
    lambda_sc = 1.0 / 100.0
    t_max = 500.0
    paramsBuffer.targetIdx = 1
    paramsBuffer.scatterCoefficient = lambda_sc
    paramsBuffer.maxTime = t_max
    paramsBuffer.lowerBBoxCorner = vec3(x=-100.0, y=-100.0, z=-100.0)
    paramsBuffer.upperBBoxCorner = vec3(x=100.0, y=100.0, z=100.0)

    # run program
    (
        hp.beginSequence()
        .And(hp.updateTensor(shadowBuffer, shadowTensor))
        .And(hp.updateTensor(paramsBuffer, paramsTensor))
        .And(hp.clearTensor(hitTensor))
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(hitTensor, hitBuffer))
        .Submit()
        .wait()
    )

    # check result
    mask = (queries["position"][:, 0] > 1.0) & (queries["position"][:, 0] < 2.0)
    n_exp = mask.sum()
    hits = hitBuffer.view
    assert hits.count == n_exp
    r = hits[: hits.count]
    # reconstruct input queries
    i_idx = (r["wavelength"][:, 0] - 200.0) / (600.0 / (N - 1))
    i_in = queries[np.round(i_idx).astype(np.int32)]
    assert np.all(i_in["wavelength"][:, 0] == r["wavelength"][:, 0])
    # check we are on local sphere coords
    r_rad = np.sqrt(np.square(r["position"]).sum(-1))
    assert np.all((r_rad <= 1.0) & (r_rad >= r_insc - 5e-7))
    dir_exp = np.array([-1.0, 0.0, 0.0])
    assert np.allclose(r["direction"], dir_exp)
    # on unit sphere, normal and position are the same
    # since we have an approximated sphere, the error is a bit larger
    assert np.abs(r["position"] - r["normal"]).max() < 5e-3
    dist = 5.0 - r["position"][:, 0]
    t_exp = dist[:, None] / i_in["vg"] + i_in["time"]
    assert np.allclose(t_exp, r["time"])
    # TODO: check throughput
