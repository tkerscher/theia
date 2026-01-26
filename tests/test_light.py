import pytest

import numpy as np
from hephaistos.pipeline import Pipeline, PipelineScheduler, runPipeline
from hephaistos.queue import dumpQueue

import theia.light
import theia.units as u
from theia.material import VACUUM_IDX
from theia.random import PhiloxRNG
from theia.ray import UnpolarizedRay
from theia.testing import BackwardLightSampler


def test_HostLightSource(rng):
    N = 32 * 256

    # create light and sampler
    ray = UnpolarizedRay()
    light = theia.light.HostLightSource(N, ray)
    sampler = theia.light.LightSampler(N, light, ray)

    # fill input buffer
    rays = light.queue.view(0)
    x = (rng.random(N) * 10.0 - 5.0) * u.m
    y = (rng.random(N) * 10.0 - 5.0) * u.m
    z = (rng.random(N) * 10.0 - 5.0) * u.m
    cos_theta = 2.0 * rng.random(N) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.random(N) * 2.0 * np.pi
    rays["position"] = np.stack([x, y, z], axis=-1)
    rays["direction"] = np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta],
        axis=-1,
    )
    lam = (rng.random((N,)) * 600.0 + 200.0) * u.nm
    rays["wavelength"] = lam
    rays["mediumIdx"] = rng.integers(0, 10, N)
    rays["time"] = rng.random((N,)) * 50.0 * u.ns
    rays["contrib"] = rng.random((N,)) + 5.0
    # run
    runPipeline(sampler.collectStages())
    # check result
    result = sampler.queue.view(0)
    assert np.allclose(result["position"], rays["position"])
    assert np.allclose(result["direction"], rays["direction"])
    assert np.allclose(result["wavelength"], rays["wavelength"])
    assert np.allclose(result["mediumIdx"], rays["mediumIdx"])
    assert np.allclose(result["time"], rays["time"])
    assert np.allclose(result["contrib"], rays["contrib"])


def test_StreamingHostLightSource(rng):
    N = 6 * 512
    batchSize = 1024

    # create input data
    def createData():
        data = {}
        data["contrib"] = 10.0 * rng.random(N)
        data["wavelength"] = 200.0 + 500.0 * rng.random(N)
        x = (rng.random(N) * 10.0 - 5.0) * u.m
        y = (rng.random(N) * 10.0 - 5.0) * u.m
        z = (rng.random(N) * 10.0 - 5.0) * u.m
        cos_theta = 2.0 * rng.random(N) - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = rng.random(N) * 2.0 * np.pi
        data["position"] = np.stack([x, y, z], axis=-1)
        dx = sin_theta * np.cos(phi)
        dy = cos_theta * np.cos(phi)
        dz = np.sin(phi)
        data["direction"] = np.stack([dx, dy, dz], axis=-1)
        data["time"] = 50.0 * rng.random(N)
        data["mediumIdx"] = rng.integers(0, 10, N)
        return data

    # create two data sets
    data1 = createData()
    data2 = createData()

    # create light and sampler
    ray = UnpolarizedRay()
    light = theia.light.StreamingHostLightSource(batchSize, ray, data=data1)
    sampler = theia.light.LightSampler(batchSize, light, ray)
    result = []

    def process(config: int, batch: int, args: None) -> None:
        result.append(dumpQueue(sampler.queue.view(config)))

    pipeline = Pipeline(sampler.collectStages())
    scheduler = PipelineScheduler(pipeline, processFn=process)
    # create tasks to process all data
    tasks = [{}, {}, {}, {"data": data2}, {}, {}]
    scheduler.schedule(tasks)
    scheduler.wait()

    # reassemble results
    fields = sampler.queue.view(0).fields
    sampled1 = {f: np.concatenate([result[i][f] for i in range(3)]) for f in fields}
    sampled2 = {f: np.concatenate([result[i][f] for i in range(3, 6)]) for f in fields}
    # checks results
    for field, values in sampled1.items():
        assert np.allclose(data1[field], values)
    for field, values in sampled2.items():
        assert np.allclose(data2[field], values)


def test_ConstWavelengthSource(rng):
    N = 32 * 256
    lam = 450.0 * u.nm

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.ConstWavelengthSource(lam)
    light = theia.light.PencilLightSource(photons, mediumIdx=VACUUM_IDX)
    sampler = theia.light.LightSampler(N, light, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())
    result = sampler.queue.view(0)

    # check result
    assert np.all(result["contrib"] == 1.0)
    assert np.all(result["wavelength"] == lam)


@pytest.mark.parametrize("normalize", [True, False])
def test_UniformWavelength(normalize: bool):
    N = 32 * 256
    lamRange, dLam = (350.0, 750.0) * u.nm, 400.0 * u.nm
    timeRange, dt = (0.0, 0.0) * u.ns, 0.0 * u.ns
    # contrib = L/p(t,lam); p(t, lam) = 1.0 / (|dLam|*|dt|)
    contrib = 1.0 if normalize else dLam

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.UniformWavelengthSource(
        lambdaRange=lamRange, normalize=normalize
    )
    light = theia.light.PencilLightSource(
        photons, timeRange=timeRange, mediumIdx=VACUUM_IDX
    )
    sampler = theia.light.LightSampler(N, light, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())
    result = sampler.queue.view(0)
    # check results
    assert np.all(result["contrib"] == contrib)
    assert np.abs(np.min(result["wavelength"]) - lamRange[0]) < 1.0
    assert np.abs(np.max(result["wavelength"]) - lamRange[1]) < 1.0


def test_FunctionWavelengthSource():
    N = 256 * 1024  # a bit more for the histogram
    lamRange = (300.0, 700.0) * u.nm

    def fn(lam: float) -> float:
        # something fancy just for testing
        x = lam / 1000
        return -x * np.log(x)

    def Fn(lam: float) -> float:
        # anti-derivative needed for testing
        x = lam / 1000
        return 250.0 * x**2 * (1.0 - 2.0 * np.log(x))

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.FunctionWavelengthSource(fn, lambdaRange=lamRange)
    light = theia.light.PencilLightSource(
        photons, timeRange=(0.0, 0.0), mediumIdx=VACUUM_IDX
    )
    sampler = theia.light.LightSampler(N, light, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())
    result = sampler.queue.view(0)

    # check result
    exp_contrib = Fn(lamRange[1]) - Fn(lamRange[0])
    assert np.allclose(result["contrib"], exp_contrib)
    assert np.abs(np.min(result["wavelength"]) - lamRange[0]) < 1.0
    assert np.abs(np.max(result["wavelength"]) - lamRange[1]) < 1.0
    # check distribution via histogram
    hist, edges = np.histogram(result["wavelength"], bins=40)
    hist = hist / N
    F = Fn(edges)
    exp_hist = (F[1:] - F[:-1]) / exp_contrib
    assert np.abs(hist - exp_hist).max() < 9e-4


def test_ConeLightSource_fwd():
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    direction = (0.8, 0.36, 0.48)
    timeRange = (10.0, 50.0) * u.ns
    opening = 0.33
    budget = 12.0

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(wavelength=100.0 * u.nm)
    light = theia.light.ConeLightSource(
        photons,
        position=position,
        direction=direction,
        timeRange=timeRange,
        cosOpeningAngle=opening,
        budget=budget,
        mediumIdx=VACUUM_IDX,
    )
    sampler = theia.light.LightSampler(N, light, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())
    result = sampler.queue.view(0)

    # check result
    assert result.count == N
    assert np.allclose(result["position"], position)
    assert (result["direction"] * direction).sum(-1).min() >= opening
    assert np.abs(np.square(result["direction"]).sum(-1) - 1.0).max() < 1e-5
    assert np.all(result["contrib"] == budget)
    assert np.all(result["mediumIdx"] == VACUUM_IDX)
    t_min, t_max = np.min(result["time"]).item(), np.max(result["time"]).item()
    assert timeRange[0] <= t_min <= timeRange[0] + 0.1
    assert timeRange[1] - 0.1 <= t_max <= timeRange[1]


def test_ConeLightSource_bwd():
    N = 32 * 256
    light_pos = (14.0, -2.0, 3.0) * u.m
    light_dir = (0.8, 0.36, 0.48)
    light_opening = 0.33
    budget = 12.0

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC01DC0FFEE)
    photon = theia.light.ConstWavelengthSource()
    light = theia.light.ConeLightSource(
        photon,
        mediumIdx=VACUUM_IDX,
        position=light_pos,
        direction=light_dir,
        timeRange=(10.0, 10.0) * u.ns,
        cosOpeningAngle=light_opening,
        budget=budget,
    )
    sampler = BackwardLightSampler(N, light, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())

    light_pos = np.array(light_pos)
    light_dir = np.array(light_dir)
    result = sampler.queue.view(0)
    # retrieve results
    startTime = result["time"]
    contrib = result["contrib"]
    # check result
    assert np.all(result["mediumIdx"] == VACUUM_IDX)
    assert np.isfinite(result["observer"]).all()
    assert np.all(result["position"] == light_pos)
    exp_dir = result["observer"] - light_pos[None, :]
    d = np.sqrt(np.square(exp_dir).sum(-1))
    exp_dir /= d[:, None]
    assert np.allclose(result["direction"], exp_dir)
    assert np.all(startTime == 10.0 * u.ns)
    cos_nrm = np.abs(np.multiply(result["normal"], result["direction"]).sum(-1))
    cos_angle = np.multiply(light_dir[None, :], result["direction"]).sum(-1)
    exp_contrib = budget * cos_nrm / ((1.0 - light_opening) * 2.0 * np.pi * d**2)
    exp_contrib = np.where(cos_angle >= light_opening, exp_contrib, 0.0)
    assert np.allclose(contrib, exp_contrib)


def test_PencilLightSource():
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    direction = (0.8, 0.36, 0.48)  # unit
    timeRange = (10.0, 50.0) * u.ns
    budget = 12.0

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(wavelength=100.0 * u.nm)
    light = theia.light.PencilLightSource(
        photons,
        position=position,
        direction=direction,
        timeRange=timeRange,
        budget=budget,
        mediumIdx=VACUUM_IDX,
    )
    sampler = theia.light.LightSampler(N, light, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())
    result = sampler.queue.view(0)

    # check result
    assert result.count == N
    assert np.allclose(result["position"], position)
    assert np.allclose(result["direction"], direction)
    assert np.all(result["contrib"] == budget)
    assert np.all(result["mediumIdx"] == VACUUM_IDX)
    t_min, t_max = np.min(result["time"]).item(), np.max(result["time"]).item()
    assert timeRange[0] <= t_min <= timeRange[0] + 0.1
    assert timeRange[1] - 0.1 <= t_max <= timeRange[1]


def test_SphericalLightSource_fwd():
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    timeRange = (10.0, 50.0) * u.ns
    budget = 12.0

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(wavelength=100.0 * u.nm)
    light = theia.light.SphericalLightSource(
        photons,
        position=position,
        timeRange=timeRange,
        budget=budget,
        mediumIdx=VACUUM_IDX,
    )
    sampler = theia.light.LightSampler(N, light, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())
    result = sampler.queue.view(0)

    # transform input to numpy array for testing
    position = np.array(position, dtype=np.float32)
    # check result
    assert result.count == N
    assert np.all(result["position"] == position)
    assert np.allclose(np.sqrt(np.square(result["direction"]).sum(-1)), 1.0)
    # uniform direction should average to zero
    assert np.abs(np.mean(result["direction"], axis=0)).max() < 0.01  # low statistics
    # check contribution
    assert np.allclose(result["contrib"], budget)
    assert np.all(result["mediumIdx"] == VACUUM_IDX)
    t_min, t_max = np.min(result["time"]).item(), np.max(result["time"]).item()
    assert timeRange[0] <= t_min <= timeRange[0] + 0.1
    assert timeRange[1] - 0.1 <= t_max <= timeRange[1]


def test_SphericalLightSource_bwd():
    N = 32 * 256
    light_pos = (14.0, -2.0, 3.0) * u.m
    budget = 1e6

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0FFEE)
    photon = theia.light.ConstWavelengthSource()
    light = theia.light.SphericalLightSource(
        photon,
        mediumIdx=VACUUM_IDX,
        position=light_pos,
        timeRange=(10.0, 10.0) * u.ns,
        budget=budget,
    )
    sampler = BackwardLightSampler(N, light, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())

    # check result
    result = sampler.queue.view(0)
    light_pos = np.array(light_pos)
    assert np.all(result["mediumIdx"] == VACUUM_IDX)
    assert np.isfinite(result["observer"]).all()
    assert np.all(result["position"] == light_pos)
    exp_dir = result["observer"] - light_pos[None, :]
    d = np.sqrt(np.square(exp_dir).sum(-1))
    exp_dir /= d[:, None]
    assert np.allclose(result["direction"], exp_dir)
    assert np.all(result["time"] == 10.0 * u.ns)
    cos_nrm = np.abs(np.multiply(result["normal"], result["direction"]).sum(-1))
    exp_contrib = budget * cos_nrm / (4.0 * np.pi * d**2)
    # bit larger error, likely precission issues (float vs double)
    assert np.allclose(result["contrib"], exp_contrib, atol=1e-5)
