import numpy as np
import hephaistos as hp

import theia.light
import theia.material
import theia.random
import theia.scene
import theia.trace

from ctypes import *
from hephaistos.glsl import mat4x3, vec2, vec3, uvec2, stackVector
from numpy.ctypeslib import as_array
from numpy.lib.recfunctions import structured_to_unstructured
from theia.util import compileShader, loadShader, packUint64


##################################### UTIL #####################################


class QueueBuffer(hp.RawBuffer):
    def __init__(self, itemType: Structure, n: int):
        self._size = 4 + n * sizeof(itemType)
        super().__init__(self._size)
        self._adr = super().address
        self._arr = (itemType * n).from_address(self._adr + 4)
        self._count = c_uint32.from_address(self._adr)
        self.count = n

    def numpy(self):
        return as_array(self._arr)

    @property
    def count(self):
        return self._count.value

    @count.setter
    def count(self, value):
        self._count.value = value


class QueueTensor(hp.ByteTensor):
    def __init__(self, itemType: Structure, n: int):
        super().__init__(4 + sizeof(itemType) * n)


class WaterModel(
    theia.material.WaterBaseModel,
    theia.material.HenyeyGreensteinPhaseFunction,
    theia.material.MediumModel,
):
    def __init__(self) -> None:
        theia.material.WaterBaseModel.__init__(self, 5.0, 1000.0, 35.0)
        theia.material.HenyeyGreensteinPhaseFunction.__init__(self, 0.6)

    ModelName = "water"


##################################### TYPES ####################################


N_PHOTONS = 4


class Photon(Structure):
    _fields_ = [
        ("wavelength", c_float),
        ("time", c_float),
        ("lin_c", c_float),
        ("log_c", c_float),
        # medium constants
        ("n", c_float),
        ("vg", c_float),
        ("mu_s", c_float),
        ("mu_e", c_float),
    ]


class PhotonHit(Structure):
    _fields_ = [
        ("wavelength", c_float),
        ("time", c_float),
        ("contribution", c_float),
    ]


class Ray(Structure):
    _fields_ = [
        ("position", vec3),
        ("direction", vec3),
        ("rngIdx", c_uint32),
        ("medium", uvec2),  # uint64
        ("photons", Photon * N_PHOTONS),
    ]


class RayHit(Structure):
    _fields_ = [
        ("position", vec3),
        ("direction", vec3),
        ("normal", vec3),
        ("hits", PhotonHit * N_PHOTONS),
    ]


class InitParams(Structure):
    _fields_ = [
        # ("medium", c_uint64),
        ("medium", uvec2),
        ("count", c_uint32),
        ("rngStride", c_uint32),
    ]


class IntersectionItem(Structure):
    _fields_ = [
        ("ray", Ray),
        ("geometryIdx", c_int32),
        ("customIdx", c_int32),
        ("triangleIdx", c_int32),
        ("barys", vec2),
        ("obj2World", mat4x3),
        ("world2Obj", mat4x3),
    ]


class ShadowRayItem(Structure):
    _fields_ = [
        ("ray", Ray),
        ("dist", c_float),
    ]


class VolumeScatterItem(Structure):
    _fields_ = [("ray", Ray), ("dist", c_float)]


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


#################################### STAGES ####################################


def test_histogram(rng):
    N = 32 * 256
    N_BINS = 64
    BIN_SIZE = 2.0
    NORM = 0.3
    t0 = 30.0

    # create integrator
    integrator = theia.trace.HistogramIntegrator(
        loadShader("response.lambertian.glsl"),
        N,
        nBins=N_BINS,
        nPhotons=N_PHOTONS,
        batchSize=128,
    )
    reducer = theia.trace.HistogramReducer(nBins=N_BINS)

    # allocate memory
    queueBuffer = QueueBuffer(RayHit, N)
    queueTensor = QueueTensor(RayHit, N)
    histogramBuffer = hp.FloatBuffer(N_BINS)
    histogramTensor = hp.FloatTensor(N_BINS)

    # create samples
    samples = queueBuffer.numpy()
    phi_dir = rng.random(N) * 2.0 * np.pi
    phi_nrm = rng.random(N) * 2.0 * np.pi
    ct_dir = rng.random(N) - 1.0
    ct_nrm = rng.random(N)  # opposite direction
    st_dir = np.sqrt(1.0 - ct_dir ** 2)
    st_nrm = np.sqrt(1.0 - ct_nrm ** 2)
    samples["position"]["x"] = rng.normal(0.0, 10.0, N)
    samples["position"]["y"] = rng.normal(0.0, 10.0, N)
    samples["position"]["z"] = rng.normal(0.0, 10.0, N)
    samples["direction"] = stackVector(
        [st_dir * np.cos(phi_dir), st_dir * np.sin(phi_dir), ct_dir], vec3
    )
    samples["normal"] = stackVector(
        [st_nrm * np.cos(phi_nrm), st_nrm * np.sin(phi_nrm), ct_nrm], vec3
    )
    contribution = rng.random(N * N_PHOTONS).reshape((N, N_PHOTONS)) * 5.0 - 2.0
    time = rng.random(N * N_PHOTONS).reshape((N, N_PHOTONS)) * 200.0
    samples["hits"]["contribution"] = contribution
    samples["hits"]["time"] = time

    # run integrator
    (
        hp.beginSequence()
        .And(hp.updateTensor(queueBuffer, queueTensor))
        .Then(integrator.integrate(queueTensor, t0, BIN_SIZE, 5))
        .Then(
            reducer.reduce(
                histogramTensor, integrator.histograms, integrator.nHist, NORM
            )
        )
        .Then(hp.retrieveTensor(histogramTensor, histogramBuffer))
        .Submit()
        .wait()
    )

    # calculate expected result
    cosine = -np.multiply(
        structured_to_unstructured(samples["normal"]),
        structured_to_unstructured(samples["direction"]),
    ).sum(-1)
    weights = (contribution * cosine[:, None] * NORM).flatten()
    hist_exp, _ = np.histogram(
        time.flatten(), N_BINS, (t0, N_BINS * BIN_SIZE + t0), weights=weights
    )
    # check result
    hist = histogramBuffer.numpy()
    assert np.allclose(hist, hist_exp)


def test_wavefront_trace(rng):
    N = 32 * 256
    RNG_BATCH = 4 * 16
    # create light
    light = theia.light.HostLightSource(N, nPhotons=4)
    # create programs
    philox = theia.random.PhiloxRNG(N, RNG_BATCH, key=0xC0110FFC0FFEE)
    wavefront_trace = hp.Program(compileShader("wavefront.trace.glsl"))
    wavefront_intersect = hp.Program(compileShader("wavefront.intersection.glsl"))

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

    # reserve memory
    rayBufferIn = QueueBuffer(Ray, N)
    rayBufferOut = QueueBuffer(Ray, N)
    rayTensor = QueueTensor(Ray, N)
    intBuffer = QueueBuffer(IntersectionItem, N)
    intTensor = QueueTensor(IntersectionItem, N)
    volBuffer = QueueBuffer(VolumeScatterItem, N)
    volTensor = QueueTensor(VolumeScatterItem, N)
    responseBuffer = QueueBuffer(RayHit, N)
    responseTensor = QueueTensor(RayHit, N)
    paramsBuffer = hp.StructureBuffer(TraceParams)
    paramsTensor = hp.StructureTensor(TraceParams)
    # bind memory
    wavefront_trace.bindParams(
        RayQueue=rayTensor,
        IntersectionQueue=intTensor,
        VolumeScatterQueue=volTensor,
        RNGBuffer=philox.tensor,
        tlas=scene.tlas,
        Params=paramsTensor,
    )
    wavefront_intersect.bindParams(
        IntersectionQueue=intTensor,
        RayQueue=rayTensor,
        ResponseQueue=responseTensor,
        Geometries=scene.geometries,
        Params=paramsTensor,
    )

    # fill input buffer
    queries = light.numpy()
    theta = rng.random(N) * np.pi
    phi = rng.random(N) * 2.0 * np.pi
    queries["direction"] = stackVector(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), vec3
    )
    queries["position"]["x"] = 10.0
    queries["position"]["y"] = 5.0
    queries["position"]["z"] = 0.0
    pos = np.array([10.0, 5.0, 0.0])
    queries["photons"]["wavelength"] = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    # Use first wavelength as constant for reordering results
    queries["photons"]["wavelength"][:, 0] = np.linspace(200.0, 800.0, N)
    queries["photons"]["log_contrib"] = rng.random((N, N_PHOTONS)) * 5.0 - 4.0
    queries["photons"]["startTime"] = rng.random((N, N_PHOTONS)) * 50.0
    queries["photons"]["lin_contrib"] = rng.random((N, N_PHOTONS))
    # upload queries
    light.update()
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
        .And(philox.dispatchNext())
        .And(light.sample(N, rayTensor, 0, media["water"], RNG_BATCH))
        .Then(hp.retrieveTensor(rayTensor, rayBufferIn))
        .Then(wavefront_trace.dispatch(N // 32))
        .Then(hp.retrieveTensor(intTensor, intBuffer))
        .And(hp.retrieveTensor(volTensor, volBuffer))
        .And(hp.clearTensor(rayTensor))
        .Then(wavefront_intersect.dispatch(N // 32))
        .Then(hp.retrieveTensor(responseTensor, responseBuffer))
        .And(hp.retrieveTensor(rayTensor, rayBufferOut))
        .Submit()
        .wait()
    )

    # check result
    r_in = rayBufferIn.numpy()
    assert intBuffer.count + volBuffer.count == N  # all items processed?
    assert rayBufferOut.count <= intBuffer.count

    # check volume items did not pass through geometry
    v = volBuffer.numpy()[: volBuffer.count]
    v_pos = structured_to_unstructured(v["ray"]["position"])
    v_dir = structured_to_unstructured(v["ray"]["direction"])
    cos_theta = np.abs(v_dir[:, 2])  # dot with +/- e_z
    cos_theta_min = np.sqrt(1.0 - (r_insc / (r_insc + d)) ** 2)  # tangent angle
    may_hit = cos_theta >= cos_theta_min + 0.01  # acount for err in geometry too
    cos_theta = cos_theta[may_hit]
    t0 = (r_insc + d) * cos_theta - np.sqrt(
        (r_insc + d) ** 2 * cos_theta ** 2 - (2 * r_insc * d + d ** 2)
    )
    # bit tricky to test, since our sphere is approximated with triangles
    assert (v["dist"][may_hit] < t0).sum() / may_hit.sum() > 0.95
    # limit outliers
    assert ((v["dist"][may_hit] - t0) / t0).max() < 0.1

    # check intersections
    i = rayBufferOut.numpy()[: rayBufferOut.count]
    # reconstruct input queries
    i_idx = (i["photons"]["wavelength"][:, 0] - 200.0) / (600.0 / (N - 1))
    i_in = r_in[np.round(i_idx).astype(np.int32)]
    # check position: direction match
    i_pos = structured_to_unstructured(i["position"])
    i_dir = i_pos - structured_to_unstructured(i_in["position"])
    i_dir /= np.sqrt(np.square(i_dir).sum(-1))[:, None]
    r_dir = structured_to_unstructured(i_in["direction"])
    assert np.abs(i_dir - r_dir).max() < 1e-4
    # check positions: on sphere
    center = np.array([x, y, r + d])
    i_dup = np.square(i_pos - center[None, :]).sum(-1)
    center[2] = -(r + d)
    i_d = np.square(i_pos - center[None, :]).sum(-1)
    mask = i_in["direction"]["z"] > 0.0
    i_d[mask] = i_dup[mask]
    i_d = np.sqrt(i_d)
    # on small sphere
    d_small = (i_d <= r + 5e-5) & (i_d >= r_insc + 5e-5)
    d_big = (i_d <= 2 * r + 5e-5) & (i_d >= 2 * r_insc + 5e-5)
    assert np.all(d_small | d_big)
    # check distance & throughput
    i_dist = np.sqrt(np.square(i_pos - pos[None, :]).sum(-1))
    i_t = i_dist[:, None] / i_in["photons"]["vg"] + i_in["photons"]["time"]
    assert np.allclose(i_t, i["photons"]["time"])
    i_thr = (lambda_sc - i_in["photons"]["mu_e"]) * i_dist[:, None]
    i_thr += i_in["photons"]["log_c"]
    assert np.allclose(i_thr, i["photons"]["log_c"], rtol=5e-5)
    # TODO: check reflectance
    # check direction
    i_normal = (i_pos - center[None, :]) / i_d[:, None]
    center[2] = r + d
    i_normal_up = (i_pos - center[None, :]) / i_d[:, None]
    i_normal[mask] = i_normal_up[mask]
    i_din = structured_to_unstructured(i_in["direction"])
    i_dout = (
        i_din - 2.0 * np.multiply(i_din, i_normal).sum(-1)[:, None] * i_normal
    )  # reflect
    n_err = np.abs(i_dout - structured_to_unstructured(i["direction"]))
    assert n_err.max() < 0.01  # TODO: error a bit hight

    # check detector hits
    hits = responseBuffer.numpy()[: responseBuffer.count]
    # reconstruct input queries
    hits_idx = (hits["hits"]["wavelength"][:, 0] - 200.0) / (600.0 / (N - 1))
    hits_in = r_in[np.round(hits_idx).astype(np.int32)]
    # check positions (note: unit icosphere local coordinates)
    hits_pos = structured_to_unstructured(hits["position"])
    hits_d = np.sqrt(np.square(hits_pos).sum(-1))
    assert np.all((hits_d <= 1.0 + 5e-5) & (hits_d >= 0.99547149974733 + 5e-5))
    # TODO: check distance & throughput
    # TODO: Calc linear throughput and final throughput
    # check normal (note: unit icosphere)
    hits_normal = hits_pos / hits_d[:, None]
    n_err = np.abs(hits_normal - structured_to_unstructured(hits["normal"]))
    assert n_err.max() < 0.01
    # test direction
    hits_din = structured_to_unstructured(hits_in["direction"])
    hits_dir = structured_to_unstructured(hits["direction"])
    n_err = np.abs(hits_din - hits_dir)
    assert n_err.max() < 1e-5
    # check the hits are on the lower sphere via ray direction
    assert hits_din[:, 2].max() < 0.0


def test_wavefront_volume(rng):
    N = 32 * 256
    RNG_BATCH = 4 * 16
    N_DET = 32
    # create programs
    program = hp.Program(compileShader("wavefront.volume.glsl"))

    # create medium
    model = WaterModel()
    water = model.createMedium()
    tensor, _, media = theia.material.bakeMaterials(media=[water])

    # allocate memory
    volumeQueueBuffer = QueueBuffer(VolumeScatterItem, N)
    volumeQueueTensor = QueueTensor(VolumeScatterItem, N)
    rayQueueBuffer = QueueBuffer(Ray, N)
    rayQueueTensor = QueueTensor(Ray, N)
    shadowQueueBuffer = QueueBuffer(ShadowRayItem, N)
    shadowQueueTensor = QueueTensor(ShadowRayItem, N)
    detectorBuffer = hp.ArrayBuffer(Detector, N_DET)
    detectorTensor = hp.ArrayTensor(Detector, N_DET)
    paramsBuffer = hp.StructureBuffer(TraceParams)
    paramsTensor = hp.StructureTensor(TraceParams)
    # rng
    philox = theia.random.PhiloxRNG(N, RNG_BATCH, key=0xC0110FFC0FFEE, startCount=256)
    rng_buffer = hp.FloatBuffer(N * RNG_BATCH)
    # bind memory
    program.bindParams(
        VolumeScatterQueue=volumeQueueTensor,
        RayQueue=rayQueueTensor,
        ShadowQueue=shadowQueueTensor,
        Detectors=detectorTensor,
        RNGBuffer=philox.tensor,
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
    volumeQueueBuffer.count = N_in
    v = volumeQueueBuffer.numpy()
    x = rng.random(N) * 10.0 - 5.0
    y = rng.random(N) * 10.0 - 5.0
    z = rng.random(N) * 10.0 - 5.0
    phi = rng.random(N) * 2.0 * np.pi
    cos_theta = 2.0 * rng.random(N) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    v["ray"]["position"] = stackVector((x, y, z), vec3)
    v["ray"]["direction"] = stackVector(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta), vec3
    )
    v["ray"]["rngIdx"] = np.arange(N) * RNG_BATCH + rng.integers(0, 12, N)
    v["ray"]["medium"] = packUint64(media["water"])
    v["ray"]["photons"]["wavelength"] = rng.random((N, N_PHOTONS)) * 500.0 + 200.0
    # Use first wavelength as constant for reordering results
    v["ray"]["photons"]["wavelength"][:, 0] = np.linspace(200.0, 800.0, N)
    v["ray"]["photons"]["time"] = rng.random((N, N_PHOTONS)) * 20.0
    v["ray"]["photons"]["lin_c"] = rng.random((N, N_PHOTONS))
    v["ray"]["photons"]["log_c"] = rng.random((N, N_PHOTONS)) * 5.0 - 3.0
    v["ray"]["photons"]["vg"] = rng.random((N, N_PHOTONS)) * 0.5 + 0.5
    v["ray"]["photons"]["mu_s"] = mu_s
    v["ray"]["photons"]["mu_e"] = mu_e
    v["dist"] = rng.random(N) * 100.0

    # run program
    (
        hp.beginSequence()
        .And(hp.updateTensor(volumeQueueBuffer, volumeQueueTensor))
        .And(hp.updateTensor(paramsBuffer, paramsTensor))
        .And(hp.updateTensor(detectorBuffer, detectorTensor))
        .And(hp.clearTensor(rayQueueTensor))
        .And(hp.clearTensor(shadowQueueTensor))
        .And(philox.dispatchNext())
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(rayQueueTensor, rayQueueBuffer))
        .And(hp.retrieveTensor(shadowQueueTensor, shadowQueueBuffer))
        .And(hp.retrieveTensor(philox.tensor, rng_buffer))
        .Submit()
        .wait()
    )

    # check result
    v = v[:N_in]  # throw away inputs that should not have been processed
    v_ph = v["ray"]["photons"]
    travelTime = v_ph["time"] + v["dist"][:, None] / v_ph["vg"]
    v_pos = structured_to_unstructured(v["ray"]["position"])
    v_dir = structured_to_unstructured(v["ray"]["direction"])
    pos = v_pos + v["dist"][:, None] * v_dir
    p_mask = np.all((pos >= -50.0) & (pos <= 50.0), -1)
    t_mask = np.any(travelTime <= t_max, -1)
    mask = p_mask & t_mask
    assert rayQueueBuffer.count == mask.sum()
    assert shadowQueueBuffer.count == mask.sum()
    N_res = mask.sum()

    # reconstruct item ordering
    r = rayQueueBuffer.numpy()[:N_res]
    r_idx = (r["photons"]["wavelength"][:, 0] - 200.0) / (600.0 / (N - 1))
    r = r[np.argsort(np.round(r_idx).astype(np.int32))]
    assert r.size == N_res
    s = shadowQueueBuffer.numpy()[:N_res]
    s_idx = (s["ray"]["photons"]["wavelength"][:, 0] - 200.0) / (600.0 / (N - 1))
    s = s[np.argsort(np.round(s_idx).astype(np.int32))]
    assert s.size == N_res
    # mask away inputs that should have been terminated
    v = v[mask]
    travelTime = travelTime[mask]
    pos = pos[mask]
    # shortcuts
    v_ph = v["ray"]["photons"]
    r_ph = r["photons"]
    s_ph = s["ray"]["photons"]
    # check we matched correctly
    assert np.all(r_ph["wavelength"][:, 0] == v_ph["wavelength"][:, 0])
    assert np.all(s_ph["wavelength"][:, 0] == v_ph["wavelength"][:, 0])

    # check created shadow items
    d_pos = structured_to_unstructured(det[targetIdx]["position"])
    s_pos = structured_to_unstructured(s["ray"]["position"])
    sd_dir = d_pos - s_pos
    center_dist = np.sqrt(np.square(sd_dir).sum(-1))
    sd_dir = sd_dir / center_dist[:, None]
    assert np.all(s["dist"] >= center_dist)
    edge_dist = np.sqrt(det[targetIdx]["radius"] ** 2 + center_dist ** 2)
    cos_cone = center_dist / edge_dist
    s_dir = structured_to_unstructured(s["ray"]["direction"])
    cos_theta = np.multiply(sd_dir, s_dir).sum(-1)
    assert np.all(cos_theta >= cos_cone)
    s_pos = structured_to_unstructured(s["ray"]["position"])
    assert np.abs(s_pos - pos).max() < 5e-6
    delta_rng = s["ray"]["rngIdx"] - v["ray"]["rngIdx"]
    assert delta_rng.max() == delta_rng.min()  # we have to always draw the same amount
    assert delta_rng.min() > 0  # check that we actually drew some random numbers
    assert np.abs(s_ph["time"] - travelTime).max() < 5e-5
    s_logc = (_lam - mu_e) * v["dist"][:, None] + v_ph["log_c"]
    assert np.abs(s_ph["log_c"] - s_logc).max() < 1e-4  # Why this large error?
    assert np.abs(s_ph["wavelength"] - v_ph["wavelength"]).max() < 1e-6

    # Check created ray queries
    r_pos = structured_to_unstructured(r["position"])
    assert np.abs(r_pos - pos).max() < 5e-6
    delta_rng = r["rngIdx"] - v["ray"]["rngIdx"]
    assert delta_rng.max() == delta_rng.min()  # we have to always draw the same amount
    assert delta_rng.min() > 0  # check that we actually drew some random numbers
    assert np.abs(r_ph["time"] - travelTime).max() < 5e-5
    # assert np.abs(r_ph["T_lin"] - mu_s / _lam).max() < 1e-6
    r_logc = (_lam - mu_e) * v["dist"][:, None] + v_ph["log_c"]
    assert np.abs(r_ph["log_c"] - r_logc).max() < 1e-4  # Why this large error?
    assert np.abs(r_ph["wavelength"] - v_ph["wavelength"]).max() < 1e-6

    # TODO: check lin_c (requires quite some calculations...)

    # we take a shortcut on testing the correct sampling:
    # the Henyey-Greenstein's g parameter is the expected value of 2pi*cos(theta)
    # since we IS the cosine, the mean of the cosines should be g
    # (HG is normalized to 1/2pi)
    v_dir = structured_to_unstructured(v["ray"]["direction"])
    r_dir = structured_to_unstructured(r["direction"])
    cos_theta = np.multiply(v_dir, r_dir).sum(-1)
    g_mc = cos_theta.mean()
    # give a bit more slack since we dont have that many samples
    assert np.abs(g_mc - model.g) < 6e-3

    # check if scattered phi is truly random
    # create a local euclidean cosy from input direction as one axis
    # the ratio of the projection onto the other two axis is the tan
    # since we expect uniform, we are free to rotate around the input direction
    # See prbt v4: chapter 3.3.3
    s = np.copysign(1.0, v_dir[:, 2])
    a = -1.0 / (s + v_dir[:, 2])
    b = a * v_dir[:, 0] * v_dir[:, 1]
    x_ax = np.stack([1.0 + s * v_dir[:, 0] ** 2 * a, s * b, -s * v_dir[:, 0]], -1)
    y_ax = np.stack([b, s + v_dir[:, 1] ** 2 * a, -v_dir[:, 1]], -1)
    # normalize to be safe
    x_ax = x_ax / np.sqrt(np.square(x_ax).sum(-1))[:, None]
    y_ax = y_ax / np.sqrt(np.square(y_ax).sum(-1))[:, None]
    # project onto axis
    x = np.multiply(x_ax, r_dir).sum(-1)
    y = np.multiply(y_ax, r_dir).sum(-1)
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
    program = hp.Program(compileShader("wavefront.shadow.glsl"))

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
    shadowBuffer = QueueBuffer(ShadowRayItem, N)
    shadowTensor = QueueTensor(ShadowRayItem, N)
    responseBuffer = QueueBuffer(RayHit, N)
    responseTensor = QueueTensor(RayHit, N)
    paramsBuffer = hp.StructureBuffer(TraceParams)
    paramsTensor = hp.StructureTensor(TraceParams)
    # bind params
    program.bindParams(
        ShadowQueue=shadowTensor,
        ResponseQueue=responseTensor,
        Geometries=scene.geometries,
        tlas=scene.tlas,
        Params=paramsTensor,
    )

    # fill input buffer
    queries = shadowBuffer.numpy()
    queries["dist"] = 50.0
    queries["ray"]["photons"]["wavelength"] = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    # Use first wavelength as constant for reordering results
    queries["ray"]["photons"]["wavelength"][:, 0] = np.linspace(200.0, 800.0, N)
    queries["ray"]["photons"]["vg"] = rng.random((N, N_PHOTONS)) * 0.5 + 0.1
    queries["ray"]["photons"]["mu_e"] = rng.random((N, N_PHOTONS)) * 0.128
    queries["ray"]["photons"]["time"] = rng.random((N, N_PHOTONS)) * 50.0
    queries["ray"]["photons"]["lin_c"] = rng.random((N, N_PHOTONS))
    # all point in z direction
    queries["ray"]["direction"]["z"] = 1.0
    # swoop in x direction
    queries["ray"]["position"]["x"] = np.linspace(0.0, 7.0, N)
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
        .And(hp.clearTensor(responseTensor))
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(responseTensor, responseBuffer))
        .Submit()
        .wait()
    )

    # check result
    mask = (queries["ray"]["position"]["x"] > 1.0) & (
        queries["ray"]["position"]["x"] < 2.0
    )
    n_exp = mask.sum()
    assert responseBuffer.count == n_exp
    r = responseBuffer.numpy()[: responseBuffer.count]
    # reconstruct input queries
    i_idx = (r["hits"]["wavelength"][:, 0] - 200.0) / (600.0 / (N - 1))
    i_in = queries[np.round(i_idx).astype(np.int32)]
    i_ph = i_in["ray"]["photons"]
    assert np.all(i_ph["wavelength"][:, 0] == r["hits"]["wavelength"][:, 0])
    r_pos = structured_to_unstructured(r["position"])
    # check we are on local sphere coords
    r_rad = np.sqrt(np.square(r_pos).sum(-1))
    assert np.all((r_rad <= 1.0) & (r_rad >= r_insc - 5e-7))
    dir_exp = np.array([-1.0, 0.0, 0.0])
    r_dir = structured_to_unstructured(r["direction"])
    assert np.allclose(r_dir, dir_exp)
    # on unit sphere, normal and position are the same
    # since we have an approximated sphere, the error is a bit larger
    r_nrm = structured_to_unstructured(r["normal"])
    assert np.abs(r_pos - r_nrm).max() < 5e-3
    dist = 5.0 - r["position"]["x"]
    t_exp = dist[:, None] / i_ph["vg"] + i_ph["time"]
    assert np.allclose(t_exp, r["hits"]["time"])
    # TODO: check throughput
