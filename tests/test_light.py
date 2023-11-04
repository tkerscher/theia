import numpy as np
import hephaistos as hp

import theia.light
import theia.material
import theia.random

from ctypes import *
from hephaistos.glsl import vec3, uvec2, stackVector
from numpy.ctypeslib import as_array
from numpy.lib.recfunctions import structured_to_unstructured
from theia.util import packUint64


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


class Ray(Structure):
    _fields_ = [
        ("position", vec3),
        ("direction", vec3),
        ("rngIdx", c_uint32),
        ("medium", uvec2),  # uint64
        ("photons", Photon * N_PHOTONS),
    ]


def test_lightsource(rng):
    # create host light
    N = 32 * 256
    RNG_STRIDE = 16
    light = theia.light.HostLightSource(N, nPhotons=N_PHOTONS)

    # create medium
    model = WaterModel()
    water = model.createMedium()
    tensor, _, media = theia.material.bakeMaterials(media=[water])

    # allocate queue
    outputBuffer = QueueBuffer(Ray, N)
    outputTensor = QueueTensor(Ray, N)

    # fill input buffer
    rays = light.numpy()
    x = rng.random(N) * 10.0 - 5.0
    y = rng.random(N) * 10.0 - 5.0
    z = rng.random(N) * 10.0 - 5.0
    cos_theta = 2.0 * rng.random(N) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    phi = rng.random(N) * 2.0 * np.pi
    rays["position"] = stackVector((x, y, z), vec3)
    rays["direction"] = stackVector(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], vec3
    )
    rays["photons"]["wavelength"] = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    rays["photons"]["startTime"] = rng.random((N, N_PHOTONS)) * 50.0
    rays["photons"]["lin_contrib"] = rng.random((N, N_PHOTONS)) + 1.0
    rays["photons"]["log_contrib"] = rng.random((N, N_PHOTONS)) * 5.0 - 4.0
    # upload input
    light.update()

    # run
    (
        hp.beginSequence()
        .And(light.sample(N, outputTensor, 0, media["water"], RNG_STRIDE))
        .Then(hp.retrieveTensor(outputTensor, outputBuffer))
        .Submit()
        .wait()
    )

    # check result
    lam = rays["photons"]["wavelength"]
    assert outputBuffer.count == N
    result = outputBuffer.numpy()
    assert np.all(result["position"] == rays["position"])
    assert np.all(result["direction"] == rays["direction"])
    assert np.all(result["rngIdx"] == np.arange(N) * RNG_STRIDE)
    water_packed = packUint64(media["water"])
    assert np.all(result["medium"]["x"] == water_packed.x)
    assert np.all(result["medium"]["y"] == water_packed.y)
    photons = result["photons"]
    assert np.all(photons["wavelength"] == lam)
    assert np.all(photons["time"] == rays["photons"]["startTime"])
    assert np.all(photons["lin_c"] == rays["photons"]["lin_contrib"])
    assert np.all(photons["log_c"] == rays["photons"]["log_contrib"])
    assert np.abs(photons["n"] - model.refractive_index(lam)).max() < 5e-6
    assert np.abs(photons["vg"] - model.group_velocity(lam)).max() < 1e-6
    assert np.abs(photons["mu_s"] - model.scattering_coef(lam)).max() < 5e-5
    mu_e = model.scattering_coef(lam) + model.absorption_coef(lam)
    assert np.abs((photons["mu_e"] - mu_e) / mu_e).max() < 4e-3


def test_sphericalLight():
    N = 32 * 256
    RNG_STRIDE = 16
    position = (14.0, -2.0, 3.0)
    lamRange, dLam = (350.0, 750.0), 400.0
    timeRange, dt = (20.0, 70.0), 50.0
    intensity = 8.0
    contrib = intensity / dLam / dt
    light = theia.light.SphericalLightSource(
        nPhotons=N_PHOTONS,
        position=position,
        wavelengthRange=lamRange,
        timeRange=timeRange,
        intensity=intensity,
    )

    # create rng
    philox = theia.random.PhiloxRNG(N, RNG_STRIDE, key=0xC0110FFC0FFEE)

    # create medium
    model = WaterModel()
    water = model.createMedium()
    tensor, _, media = theia.material.bakeMaterials(media=[water])

    # allocate queue
    outputBuffer = QueueBuffer(Ray, N)
    outputTensor = QueueTensor(Ray, N)

    # run
    (
        hp.beginSequence()
        .And(philox.dispatchNext())
        .Then(light.sample(N, outputTensor, philox.tensor, media["water"], RNG_STRIDE))
        .Then(hp.retrieveTensor(outputTensor, outputBuffer))
        .Submit()
        .wait()
    )

    # check result
    assert outputBuffer.count == N
    result = outputBuffer.numpy()
    photons = result["photons"]
    pos = structured_to_unstructured(result["position"])
    assert np.all(pos == position)
    # uniform direction should average to zero
    dir = structured_to_unstructured(result["direction"])
    assert np.abs(np.mean(dir, axis=0)).max() < 0.01  # low statistics
    # correct rng indices
    assert np.all(np.diff(result["rngIdx"]) == RNG_STRIDE)
    assert np.all(
        np.mod(result["rngIdx"], RNG_STRIDE, dtype=np.int32) == light.rngSamples
    )
    # check medium
    water_packed = packUint64(media["water"])
    assert np.all(result["medium"]["x"] == water_packed.x)
    assert np.all(result["medium"]["y"] == water_packed.y)
    # check contribution
    assert np.all(photons["log_c"] == 0.0)
    assert np.all(photons["lin_c"] == contrib)
    # lazily check "uniform" in time and lambda: check min/max
    assert np.abs(np.min(photons["time"]) - timeRange[0]) < 1.0
    assert np.abs(np.max(photons["time"]) - timeRange[1]) < 1.0
    assert np.abs(np.min(photons["wavelength"]) - lamRange[0]) < 1.0
    assert np.abs(np.max(photons["wavelength"]) - lamRange[1]) < 1.0
    # check constants
    lam = photons["wavelength"].flatten()
    assert np.abs(photons["n"].flatten() - model.refractive_index(lam)).max() < 5e-6
    assert np.abs(photons["vg"].flatten() - model.group_velocity(lam)).max() < 1e-6
    assert np.abs(photons["mu_s"].flatten() - model.scattering_coef(lam)).max() < 5e-5
    mu_e = model.scattering_coef(lam) + model.absorption_coef(lam)
    assert np.abs((photons["mu_e"].flatten() - mu_e) / mu_e).max() < 4e-3
