import numpy as np

import theia.light
import theia.items
import theia.material
import theia.random

from ctypes import *

from common.models import WaterModel

from hephaistos.queue import QueueTensor, as_queue
from hephaistos.pipeline import RetrieveTensorStage, runPipeline

from theia.util import packUint64


def test_lightsource(rng):
    N = 32 * 256
    N_LAMBDA = 4

    # create light & sampler
    light = theia.light.HostLightSource(N, nLambda=N_LAMBDA)
    sampler = theia.light.LightSampler(light, N)

    # fill input buffer
    rays = light.view(0)
    x = rng.random(N) * 10.0 - 5.0
    y = rng.random(N) * 10.0 - 5.0
    z = rng.random(N) * 10.0 - 5.0
    cos_theta = 2.0 * rng.random(N) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.random(N) * 2.0 * np.pi
    rays["position"] = np.stack([x, y, z], axis=-1)
    rays["direction"] = np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta],
        axis=-1,
    )
    lam = rng.random((N, N_LAMBDA)) * 600.0 + 200.0
    rays["wavelength"] = lam
    rays["startTime"] = rng.random((N, N_LAMBDA)) * 50.0
    rays["contrib"] = rng.random((N, N_LAMBDA)) + 1.0

    # run
    runPipeline([light, sampler])
    result = sampler.view(0)

    # check result
    assert result.count == N
    assert np.allclose(result["position"], rays["position"])
    assert np.allclose(result["direction"], rays["direction"])
    assert np.allclose(result["wavelength"], lam)
    assert np.allclose(result["startTime"], rays["startTime"])
    assert np.allclose(result["contrib"], rays["contrib"])


def test_sphericalLight():
    N = 32 * 256
    N_LAMBDA = 4
    position = (14.0, -2.0, 3.0)
    lamRange, dLam = (350.0, 750.0), 400.0
    timeRange, dt = (20.0, 70.0), 50.0
    intensity = 8.0
    contrib = intensity / dLam / dt

    # create light
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    light = theia.light.SphericalLightSource(
        nLambda=N_LAMBDA,
        position=position,
        lambdaRange=lamRange,
        timeRange=timeRange,
        intensity=intensity,
    )
    sampler = theia.light.LightSampler(light, N, rng=philox)
    # run
    runPipeline([philox, light, sampler])
    result = sampler.view(0)

    # check result
    assert result.count == N
    assert np.all(result["position"] == position)
    # uniform direction should average to zero
    assert np.abs(np.mean(result["direction"], axis=0)).max() < 0.01  # low statistics
    # check contribution
    assert np.all(result["contrib"] == contrib)
    # lazily check "uniform" in time and lambda: check min/max
    assert np.abs(np.min(result["startTime"]) - timeRange[0]) < 1.0
    assert np.abs(np.max(result["startTime"]) - timeRange[1]) < 1.0
    assert np.abs(np.min(result["wavelength"]) - lamRange[0]) < 1.0
    assert np.abs(np.max(result["wavelength"]) - lamRange[1]) < 1.0
