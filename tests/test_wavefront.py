import numpy as np
import hephaistos as hp
import theia.material
import theia.random
from ctypes import *
from hephaistos.glsl import vec3, uvec2, stackVector
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


class RayItem(Structure):
    _fields_ = [("ray", Ray), ("targetIdx", c_int32)]


class VolumeScatterItem(Structure):
    _fields_ = [("ray", Ray), ("targetIdx", c_int32), ("dist", c_float)]


class TraceParams(Structure):
    _fields_ = [
        ("sampleScatteringLength", c_float),
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
        # .And(hp.updateTensor(paramBuffer, paramTensor))
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
    paramsBuffer.sampleScatteringLength = _lam
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
