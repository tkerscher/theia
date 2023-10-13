import numpy as np
import hephaistos as hp
import theia.material
import theia.random
import theia.scene
from ctypes import *
from hephaistos.glsl import mat4x3, vec2, vec3, uvec2, stackVector
from numpy.ctypeslib import as_array
from numpy.lib.recfunctions import structured_to_unstructured
from theia.util import compileShader, packUint64


##################################### UTIL #####################################


class QueueBuffer(hp.RawBuffer):
    def __init__(self, itemType: Structure, n: int):
        self._size = 4 + n * sizeof(itemType)
        super().__init__(self._size)
        self._adr = super().address
        self._arr = (itemType * n).from_address(self._adr + 4)
        self._count = c_uint32.from_address(self._adr)

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
        ("travelTime", c_float),
        ("log_radiance", c_float),
        ("T_lin", c_float),
        ("T_log", c_float),
        # medium constants
        ("n", c_float),
        ("vg", c_float),
        ("mu_s", c_float),
        ("mu_e", c_float),
    ]


class PhotonHit(Structure):
    _fields_ = [
        ("wavelength", c_float),
        ("travelTime", c_float),
        ("log_radiance", c_float),
        ("throughput", c_float)
    ]


class Ray(Structure):
    _fields_ = [
        ("position", vec3),
        ("direction", vec3),
        ("rngIdx", c_uint32),
        ("medium", uvec2),  # uint64
        ("photons", Photon * N_PHOTONS),
    ]


class PhotonQuery(Structure):
    _fields_ = [
        ("wavelength", c_float),
        ("log_radiance", c_float),
        ("startTime", c_float),
        ("probability", c_float),
    ]


class RayQuery(Structure):
    _fields_ = [
        ("position", vec3),
        ("direction", vec3),
        ("targetIdx", c_int32),
        ("photons", PhotonQuery * N_PHOTONS),
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
        ("targetIdx", c_int32),
        ("geometryIdx", c_int32),
        ("customIdx", c_int32),
        ("triangleIdx", c_int32),
        ("barys", vec2),
        ("obj2World", mat4x3),
        ("world2Obj", mat4x3)
    ]


class ResponseItem(Structure):
    _fields_ = [
        ("position", vec3),
        ("direction", vec3),
        ("normal", vec3),
        ("detectorIdx", c_int32),
        ("hits", PhotonHit * N_PHOTONS)
    ]


class RayItem(Structure):
    _fields_ = [("ray", Ray), ("targetIdx", c_int32)]


class VolumeScatterItem(Structure):
    _fields_ = [("ray", Ray), ("targetIdx", c_int32), ("dist", c_float)]


class TraceParams(Structure):
    _fields_ = [
        ("scatterCoefficient", c_float),
        ("maxTime", c_float),
        ("lowerBBoxCorner", vec3),
        ("upperBBoxCorner", vec3),
    ]


#################################### STAGES ####################################


def test_wavefront_init(rng):
    N = 32 * 256
    RNG_STRIDE = 16
    # create program
    program = hp.Program(compileShader("wavefront.init.glsl"))

    # create medium
    model = WaterModel()
    water = model.createMedium()
    tensor, _, media = theia.material.bakeMaterials(media=[water])

    # allocate memory
    inputBuffer = hp.ArrayBuffer(RayQuery, N)
    inputTensor = hp.ArrayTensor(RayQuery, N)
    outputBuffer = QueueBuffer(RayItem, N)
    outputTensor = QueueTensor(RayItem, N)
    # bind memory
    program.bindParams(QueryInput=inputTensor, RayQueue=outputTensor)

    # fill input buffer
    queries = inputBuffer.numpy()
    x = rng.random(N) * 10.0 - 5.0
    y = rng.random(N) * 10.0 - 5.0
    z = rng.random(N) * 10.0 - 5.0
    theta = rng.random(N) * np.pi
    phi = rng.random(N) * 2.0 * np.pi
    queries["position"] = stackVector((x, y, z), vec3)
    queries["direction"] = stackVector(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), vec3
    )
    queries["targetIdx"] = rng.integers(1, 256, N)
    queries["photons"]["wavelength"] = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    queries["photons"]["log_radiance"] = rng.random((N, N_PHOTONS)) * 5.0 - 4.0
    queries["photons"]["startTime"] = rng.random((N, N_PHOTONS)) * 50.0
    queries["photons"]["probability"] = rng.random((N, N_PHOTONS))
    # create push descriptor
    push = InitParams(count=N, medium=packUint64(media["water"]), rngStride=RNG_STRIDE)

    # run program
    (
        hp.beginSequence()
        .And(hp.updateTensor(inputBuffer, inputTensor))
        .Then(program.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(outputTensor, outputBuffer))
        .Submit()
        .wait()
    )

    # check result
    lam = queries["photons"]["wavelength"]
    assert outputBuffer.count == N
    result = outputBuffer.numpy()
    assert np.all(result["targetIdx"] == queries["targetIdx"])
    assert np.all(result["ray"]["position"] == queries["position"])
    assert np.all(result["ray"]["direction"] == queries["direction"])
    assert np.all(result["ray"]["rngIdx"] == np.arange(N) * RNG_STRIDE)
    water_packed = packUint64(media["water"])
    assert np.all(result["ray"]["medium"]["x"] == water_packed.x)
    assert np.all(result["ray"]["medium"]["y"] == water_packed.y)
    photons = result["ray"]["photons"]
    assert np.all(photons["wavelength"] == lam)
    assert np.all(photons["travelTime"] == queries["photons"]["startTime"])
    assert np.all(photons["log_radiance"] == queries["photons"]["log_radiance"])
    assert np.all(photons["T_lin"] == queries["photons"]["probability"])
    assert np.all(photons["T_log"] == 0.0)
    assert np.abs(photons["n"] - model.refractive_index(lam)).max() < 5e-6
    assert np.abs(photons["vg"] - model.group_velocity(lam)).max() < 1e-6
    assert np.abs(photons["mu_s"] - model.scattering_coef(lam)).max() < 5e-5
    mu_e = model.scattering_coef(lam) + model.absorption_coef(lam)
    assert np.abs((photons["mu_e"] - mu_e) / mu_e).max() < 4e-3


def test_wavefront_trace(rng):
    N = 32 * 256
    RNG_BATCH = 4 * 16
    # create programs
    philox = theia.random.PhiloxRNG(N, RNG_BATCH, key=0xC0110FFC0FFEE)
    wavefront_init = hp.Program(compileShader("wavefront.init.glsl"))
    wavefront_trace = hp.Program(compileShader("wavefront.trace.glsl"))
    wavefront_intersect = hp.Program(compileShader("wavefront.intersection.glsl"))

    # create material
    water = WaterModel().createMedium()
    glass = theia.material.BK7Model().createMedium()
    mat = theia.material.Material("mat", glass, water, flags=theia.material.Material.TARGET_BIT)
    tensor, material, media = theia.material.bakeMaterials(mat)

    # create scene
    store = theia.scene.MeshStore({
        "cube" : "assets/cone.stl",
        "sphere" : "assets/sphere.stl"
    })
    r, d = 50.0, 3.0
    r_insc = r * 0.99547149974733 # radius of inscribed sphere (icosphere)
    x, y, z = 10., 5.0, 0.0
    t1 = theia.scene.Transform.Scale(r, r, r).translate(x, y, z + r + d)
    c1 = store.createInstance("sphere", "mat", transform=t1, detectorId=0)
    t2 = theia.scene.Transform.Scale(r, r, r).translate(x, y , z - r - d)
    c2 = store.createInstance("sphere", "mat", transform=t2, detectorId=1)
    scene = theia.scene.Scene([c1,c2], material, media["water"])

    # reserve memory
    inputBuffer = hp.ArrayBuffer(RayQuery, N)
    inputTensor = hp.ArrayTensor(RayQuery, N)
    rayBufferIn = QueueBuffer(RayItem, N)
    rayBufferOut = QueueBuffer(RayItem, N)
    rayTensor = QueueTensor(RayItem, N)
    intBuffer = QueueBuffer(IntersectionItem, N)
    intTensor = QueueTensor(IntersectionItem, N)
    volBuffer = QueueBuffer(VolumeScatterItem, N)
    volTensor = QueueTensor(VolumeScatterItem, N)
    responseBuffer = QueueBuffer(ResponseItem, N)
    responseTensor = QueueTensor(ResponseItem, N)
    paramsBuffer = hp.StructureBuffer(TraceParams)
    paramsTensor = hp.StructureTensor(TraceParams)
    # bind memory
    wavefront_init.bindParams(QueryInput=inputTensor, RayQueue=rayTensor)
    wavefront_trace.bindParams(
        RayQueue=rayTensor,
        IntersectionQueue=intTensor,
        VolumeScatterQueue=volTensor,
        RNGBuffer=philox.tensor,
        tlas=scene.tlas,
        TraceParams=paramsTensor
    )
    wavefront_intersect.bindParams(
        IntersectionQueue=intTensor,
        RayQueue=rayTensor,
        ResponseQueue=responseTensor,
        Geometries=scene.geometries,
        TraceParams=paramsTensor
    )

    # fill input buffer
    queries = inputBuffer.numpy()
    theta = rng.random(N) * np.pi
    phi = rng.random(N) * 2.0 * np.pi
    queries["direction"] = stackVector(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), vec3
    )
    queries["targetIdx"] = 1
    queries["position"]["x"] = 10.0
    queries["position"]["y"] = 5.0
    queries["position"]["z"] = 0.0
    pos = np.array([10., 5., 0.])
    queries["photons"]["wavelength"] = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    # Use first wavelength as constant for reordering results
    queries["photons"]["wavelength"][:,0] = np.linspace(200., 800., N)
    queries["photons"]["log_radiance"] = rng.random((N, N_PHOTONS)) * 5.0 - 4.0
    queries["photons"]["startTime"] = rng.random((N, N_PHOTONS)) * 50.0
    queries["photons"]["probability"] = rng.random((N, N_PHOTONS))
    # create params
    lambda_sc = 1.0 / 100.0
    t_max = 500.0
    paramsBuffer.scatterCoefficient = lambda_sc
    paramsBuffer.maxTime = t_max
    paramsBuffer.lowerBBoxCorner = vec3(x=-100.,y=-100.,z=-100.)
    paramsBuffer.upperBBoxCorner = vec3(x=100.,y=100.,z=100.)
    # create push descriptor
    push = InitParams(count=N, medium=packUint64(media["water"]), rngStride=RNG_BATCH)

    # run program
    (
        hp.beginSequence()
        .And(hp.updateTensor(inputBuffer, inputTensor))
        .And(hp.updateTensor(paramsBuffer, paramsTensor))
        .Then(philox.dispatchNext())
        .Then(wavefront_init.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(rayTensor, rayBufferIn))
        .Then(wavefront_trace.dispatch(N // 32))
        .Then(hp.retrieveTensor(intTensor, intBuffer))
        .And(hp.retrieveTensor(volTensor, volBuffer))
        .And(hp.fillTensor(rayTensor))
        .Then(wavefront_intersect.dispatch(N // 32))
        .Then(hp.retrieveTensor(responseTensor, responseBuffer))
        .And(hp.retrieveTensor(rayTensor, rayBufferOut))
        .Submit()
        .wait()
    )

    # check result
    r_in = rayBufferIn.numpy()
    assert intBuffer.count + volBuffer.count == N # all items processed?
    assert rayBufferOut.count <= intBuffer.count

    # check volume items did not pass through geometry
    v = volBuffer.numpy()[:volBuffer.count]
    v_pos = structured_to_unstructured(v["ray"]["position"])
    v_dir = structured_to_unstructured(v["ray"]["direction"])
    cos_theta = np.abs(v_dir[:,2]) # dot with +/- e_z
    cos_theta_min = np.sqrt(1.0 - (r_insc/(r_insc+d))**2) # tangent angle
    may_hit = cos_theta >= cos_theta_min + 0.01 # acount for err in geometry too
    cos_theta = cos_theta[may_hit]
    t0 = (r_insc+d)*cos_theta-np.sqrt((r_insc+d)**2*cos_theta**2 - (2*r_insc*d+d**2))
    # bit tricky to test, since our sphere is approximated with triangles
    assert (v["dist"][may_hit] < t0).sum() / may_hit.sum() > 0.95
    # limit outliers
    assert ((v["dist"][may_hit] - t0)/t0).max() < 0.1

    # check intersections
    i = rayBufferOut.numpy()[:rayBufferOut.count]
    # reconstruct input queries
    i_idx = (i["ray"]["photons"]["wavelength"][:,0] - 200.0) / (600.0 / (N - 1))
    i_in = r_in[np.round(i_idx).astype(np.int32)]
    # check position: direction match
    i_pos = structured_to_unstructured(i["ray"]["position"])
    i_dir = i_pos - structured_to_unstructured(i_in["ray"]["position"])
    i_dir /= np.sqrt(np.square(i_dir).sum(-1))[:,None]
    r_dir = structured_to_unstructured(i_in["ray"]["direction"])
    assert np.abs(i_dir - r_dir).max() < 1e-4
    # check positions: on sphere
    center = np.array([x, y, r + d])
    i_dup = np.square(i_pos - center[None,:]).sum(-1)
    center[2] = -(r + d)
    i_d = np.square(i_pos - center[None,:]).sum(-1)
    mask = i_in["ray"]["direction"]["z"] > 0.0
    i_d[mask] = i_dup[mask]
    i_d = np.sqrt(i_d)
    assert np.all((i_d <= r + 5e-5) & (i_d >= r_insc + 5e-5))
    # check distance & throughput
    i_dist = np.sqrt(np.square(i_pos - pos[None,:]).sum(-1))
    i_t = i_dist[:,None] / i_in["ray"]["photons"]["vg"] + i_in["ray"]["photons"]["travelTime"]
    assert np.allclose(i_t, i["ray"]["photons"]["travelTime"])
    i_thr = (lambda_sc - i_in["ray"]["photons"]["mu_e"]) * i_dist[:,None]
    assert np.allclose(i_thr, i["ray"]["photons"]["T_log"])
    # TODO: check reflectance
    # check direction
    i_normal = (i_pos - center[None,:]) / i_d[:,None]
    center[2] = r + d
    i_normal_up = (i_pos - center[None,:]) / i_d[:,None]
    i_normal[mask] = i_normal_up[mask]
    i_din = structured_to_unstructured(i_in["ray"]["direction"])
    i_dout = i_din - 2.0 * np.multiply(i_din, i_normal).sum(-1)[:,None] * i_normal # reflect
    n_err = np.abs(i_dout - structured_to_unstructured(i["ray"]["direction"]))
    assert n_err.max() < 0.01 # TODO: error a bit hight

    # check detector hits
    hits = responseBuffer.numpy()[:responseBuffer.count]
    assert np.all(hits["detectorIdx"] == 1)
    # reconstruct input queries
    hits_idx = (hits["hits"]["wavelength"][:,0] - 200.0) / (600.0 / (N - 1))
    hits_in = r_in[np.round(hits_idx).astype(np.int32)]
    # check positions (note: unit icosphere)
    hits_pos = structured_to_unstructured(hits["position"])
    hits_d = np.sqrt(np.square(hits_pos).sum(-1))
    assert np.all((hits_d <= 1.0 + 5e-5) & (hits_d >= 0.99547149974733 + 5e-5))
    # TODO: check distance & throughput
    # TODO: Calc linear throughput and final throughput
    # check normal (note: unit icosphere)
    hits_normal = hits_pos / hits_d[:,None]
    n_err = np.abs(hits_normal - structured_to_unstructured(hits["normal"]))
    assert n_err.max() < 0.01
    # test direction
    hits_din = structured_to_unstructured(hits_in["ray"]["direction"])
    hits_dir = structured_to_unstructured(hits["direction"])
    n_err = np.abs(hits_din - hits_dir)
    assert n_err.max() < 1e-5


def test_wavefront_volume(rng):
    N = 32 * 256
    RNG_BATCH = 4 * 16
    # create programs
    program = hp.Program(compileShader("wavefront.volume.glsl"))

    # create medium
    model = WaterModel()
    water = model.createMedium()
    tensor, _, media = theia.material.bakeMaterials(media=[water])

    # allocate memory
    volumeQueueBuffer = QueueBuffer(VolumeScatterItem, N)
    volumeQueueTensor = QueueTensor(VolumeScatterItem, N)
    rayQueueBuffer = QueueBuffer(RayItem, N)
    rayQueueTensor = QueueTensor(RayItem, N)
    paramsBuffer = hp.StructureBuffer(TraceParams)
    paramsTensor = hp.StructureTensor(TraceParams)
    # rng
    philox = theia.random.PhiloxRNG(N, RNG_BATCH, key=0xC0110FFC0FFEE, startCount=256)
    rng_buffer = hp.FloatBuffer(N * RNG_BATCH)
    # bind memory
    program.bindParams(
        VolumeScatterQueue=volumeQueueTensor,
        RayQueue=rayQueueTensor,
        RNGBuffer=philox.tensor,
        TraceParams=paramsTensor,
    )

    # set params
    t_max = 80.0
    _lam = 10.0
    mu_s = 0.6
    mu_e = 0.9
    paramsBuffer.scatterCoefficient = _lam
    paramsBuffer.maxTime = t_max
    paramsBuffer.lowerBBoxCorner = vec3(x=-50.0, y=-50.0, z=-50.0)
    paramsBuffer.upperBBoxCorner = vec3(x=50.0, y=50.0, z=50.0)
    # fill input
    N_in = N - 146
    volumeQueueBuffer.count = N_in
    v = volumeQueueBuffer.numpy()
    x = rng.random(N) * 10.0 - 5.0
    y = rng.random(N) * 10.0 - 5.0
    z = rng.random(N) * 10.0 - 5.0
    theta = rng.random(N) * np.pi
    phi = rng.random(N) * 2.0 * np.pi
    v["ray"]["position"] = stackVector((x, y, z), vec3)
    v["ray"]["direction"] = stackVector(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), vec3
    )
    v["ray"]["rngIdx"] = np.arange(N) * RNG_BATCH + rng.integers(0, 12, N)
    v["ray"]["medium"] = packUint64(media["water"])
    v["ray"]["photons"]["wavelength"] = rng.random((N, N_PHOTONS)) * 500.0 + 200.0
    v["ray"]["photons"]["travelTime"] = rng.random((N, N_PHOTONS)) * 20.0
    v["ray"]["photons"]["log_radiance"] = rng.random((N, N_PHOTONS))
    v["ray"]["photons"]["T_lin"] = 1.0
    v["ray"]["photons"]["T_log"] = 0.0
    v["ray"]["photons"]["vg"] = rng.random((N, N_PHOTONS)) * 0.5 + 0.5
    v["ray"]["photons"]["mu_s"] = mu_s
    v["ray"]["photons"]["mu_e"] = mu_e
    v["dist"] = rng.random(N) * 100.0
    # we'll use targetIdx to rematch the result with the queries
    v["targetIdx"] = np.arange(N)

    # run program
    (
        hp.beginSequence()
        .And(hp.updateTensor(volumeQueueBuffer, volumeQueueTensor))
        .And(hp.updateTensor(paramsBuffer, paramsTensor))
        .And(philox.dispatchNext())
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(rayQueueTensor, rayQueueBuffer))
        .And(hp.retrieveTensor(philox.tensor, rng_buffer))
        .Submit()
        .wait()
    )

    # check result
    v = v[:N_in]  # throw away inputs that should not have been processed
    v_ph = v["ray"]["photons"]
    travelTime = v_ph["travelTime"] + v["dist"][:, None] / v_ph["vg"]
    v_pos = structured_to_unstructured(v["ray"]["position"])
    v_dir = structured_to_unstructured(v["ray"]["direction"])
    pos = v_pos + v["dist"][:, None] * v_dir
    p_mask = np.all((pos >= -50.0) & (pos <= 50.0), -1)
    t_mask = np.any(travelTime <= t_max, -1)
    mask = p_mask & t_mask
    assert rayQueueBuffer.count == mask.sum()
    N_res = mask.sum()

    # check if the targetIdx are genuine
    r = rayQueueBuffer.numpy()[:N_res]
    s, idx = np.unique(r["targetIdx"], return_index=True)
    assert s.size == N_res
    # sort result by targetIdx to align with query
    r = r[idx]
    # mask away inputs that should have been terminated
    v = v[mask]
    travelTime = travelTime[mask]
    pos = pos[mask]
    # sanity check
    assert np.all(r["targetIdx"] == v["targetIdx"])

    # check result
    r_ph = r["ray"]["photons"]
    v_ph = v["ray"]["photons"]
    # assert result
    r_pos = structured_to_unstructured(r["ray"]["position"])
    assert np.abs(r_pos - pos).max() < 5e-6
    delta_rng = r["ray"]["rngIdx"] - v["ray"]["rngIdx"]
    assert delta_rng.max() == delta_rng.min()  # we have to always draw the same amount
    assert delta_rng.min() > 0  # check that we actually drew some random numbers
    assert np.abs(r_ph["travelTime"] - travelTime).max() < 5e-5
    assert np.abs(r_ph["T_lin"] - mu_s / _lam).max() < 1e-6
    assert np.abs(r_ph["T_log"] - (_lam - mu_e) * v["dist"][:, None]).max() < 1e-6
    assert np.abs(r_ph["wavelength"] - v_ph["wavelength"]).max() < 1e-6
    assert np.abs(r_ph["log_radiance"] - v_ph["log_radiance"]).max() < 1e-6

    # we take a shortcut on testing the correct sampling:
    # the Henyey-Greenstein's g parameter is the expected value of 2pi*cos(theta)
    # since we IS the cosine, the mean of the cosines should be g
    # (HG is normalized to 1/2pi)
    v_dir = structured_to_unstructured(v["ray"]["direction"])
    r_dir = structured_to_unstructured(r["ray"]["direction"])
    cos_theta = np.multiply(v_dir, r_dir).sum(-1)
    g_mc = cos_theta.mean()
    # give a bit more slack since we dont have that many samples
    assert np.abs(g_mc - model.g) < 1e-3

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
