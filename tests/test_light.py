import pytest

import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats

import hephaistos as hp
from hephaistos.pipeline import Pipeline, PipelineScheduler, runPipeline
from hephaistos.queue import dumpQueue
from ctypes import Structure, c_float

import theia.light
import theia.units as u
from theia.compiler import compileShader
from theia.material import MaterialStore, PureWaterModel, VACUUM_IDX
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


def test_ConstWavelengthSource():
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


@pytest.mark.parametrize("particle", [True, False], ids=["particle", "ray"])
def test_ConeLightSource_fwd(particle: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    direction = (0.8, 0.36, 0.48)
    timeRange = (10.0, 50.0) * u.ns
    opening = 0.33
    budget = 12.0

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
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
        emitParticles=particle,
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
    assert np.all(result["mediumIdx"] == VACUUM_IDX)
    t_min, t_max = np.min(result["time"]).item(), np.max(result["time"]).item()
    assert timeRange[0] <= t_min <= timeRange[0] + 0.1
    assert timeRange[1] - 0.1 <= t_max <= timeRange[1]
    if not particle:
        assert np.all(result["contrib"] == budget)


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
        mediumIdx=VACUUM_IDX,
        position=light_pos,
        direction=light_dir,
        timeRange=(10.0, 10.0) * u.ns,
        cosOpeningAngle=light_opening,
        budget=budget,
    )
    sampler = BackwardLightSampler(N, light, ray, photon, rng=philox)
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


@pytest.mark.parametrize("particle", [True, False], ids=["particle", "ray"])
def test_PencilLightSource(particle: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    direction = (0.8, 0.36, 0.48)  # unit
    timeRange = (10.0, 50.0) * u.ns
    budget = 12.0

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    philox = PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(wavelength=100.0 * u.nm)
    light = theia.light.PencilLightSource(
        photons,
        position=position,
        direction=direction,
        timeRange=timeRange,
        budget=budget,
        mediumIdx=VACUUM_IDX,
        emitParticles=particle,
    )
    sampler = theia.light.LightSampler(N, light, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())
    result = sampler.queue.view(0)

    # check result
    assert result.count == N
    assert np.allclose(result["position"], position)
    assert np.allclose(result["direction"], direction)
    assert np.all(result["mediumIdx"] == VACUUM_IDX)
    t_min, t_max = np.min(result["time"]).item(), np.max(result["time"]).item()
    assert timeRange[0] <= t_min <= timeRange[0] + 0.1
    assert timeRange[1] - 0.1 <= t_max <= timeRange[1]
    if not particle:
        assert np.all(result["contrib"] == budget)


@pytest.mark.parametrize("particle", [True, False], ids=["particle", "ray"])
def test_SphericalLightSource_fwd(particle: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    timeRange = (10.0, 50.0) * u.ns
    budget = 12.0

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    philox = PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(wavelength=100.0 * u.nm)
    light = theia.light.SphericalLightSource(
        photons,
        position=position,
        timeRange=timeRange,
        budget=budget,
        mediumIdx=VACUUM_IDX,
        emitParticles=particle,
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
    assert np.all(result["mediumIdx"] == VACUUM_IDX)
    t_min, t_max = np.min(result["time"]).item(), np.max(result["time"]).item()
    assert timeRange[0] <= t_min <= timeRange[0] + 0.1
    assert timeRange[1] - 0.1 <= t_max <= timeRange[1]
    if not particle:
        assert np.allclose(result["contrib"], budget)
    # uniform direction should average to zero
    assert np.abs(np.mean(result["direction"], axis=0)).max() < 0.012  # low statistics


def test_SphericalLightSource_bwd():
    N = 32 * 256
    light_pos = (14.0, -2.0, 3.0) * u.m
    budget = 1e6

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0FFEE)
    photon = theia.light.ConstWavelengthSource()
    light = theia.light.SphericalLightSource(
        mediumIdx=VACUUM_IDX,
        position=light_pos,
        timeRange=(10.0, 10.0) * u.ns,
        budget=budget,
    )
    sampler = BackwardLightSampler(N, light, ray, photon, rng=philox)
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


@pytest.mark.parametrize(
    "count,frankTamm,particle",
    [
        (True, True, False),
        (True, False, True),
        (False, False, False),
        (False, True, False),
    ],
)
def test_CherenkovLightSource_fwd(count: bool, frankTamm: bool, particle: bool):
    N = 32 * 256
    dir = np.array([0.36, 0.48, 0.8])
    dist = 200.0 * u.m
    startPos, startT = -dir * 0.5 * dist, -0.5 * dist / u.c
    endPos, endT = dir * 0.5 * dist, 0.5 * dist / u.c

    # build media
    model = PureWaterModel()
    water = model.createMedium(name="water")
    store = MaterialStore([], media=[water])
    waterIdx = store.media["water"]
    # build pipeline
    ray = UnpolarizedRay(particle=particle)
    philox = PhiloxRNG(key=0xC0FFEE)
    photon = theia.light.UniformWavelengthSource()
    source = theia.light.CherenkovLightSource(
        photon,
        mediumIdx=waterIdx,
        trackStart=startPos,
        trackEnd=endPos,
        startTime=startT,
        endTime=endT,
        usePhotonCount=count,
        applyFrankTamm=frankTamm,
        emitParticles=particle,
    )
    sampler = theia.light.LightSampler(N, source, ray, rng=philox, materials=store)
    # run
    runPipeline(sampler.collectStages())
    hp.checkCurrentDeviceHealth()
    samples = sampler.queue.view(0)

    # check result
    distA = np.sqrt(np.square(samples["position"] - startPos).sum(-1))
    distB = np.sqrt(np.square(samples["position"] - endPos).sum(-1))
    assert np.allclose(distA + distB, dist)  # check position is on track
    t = samples["time"].ravel()
    assert np.all((t >= startT) & (t <= endT))
    l = distA / dist
    t_exp = (1.0 - l) * startT + l * endT
    assert np.allclose(t, t_exp)
    cos_theta = np.multiply(samples["direction"], dir[None, :]).sum(-1)
    lam = samples["wavelength"]
    n = model.refractive_index(lam)
    assert np.allclose(cos_theta, 1.0 / n)
    if not particle:
        # calculate expected contrib
        contrib = np.ones_like(samples["contrib"]) * dist
        if frankTamm:
            contrib *= theia.light.frankTamm(lam, n, usePhotonCount=count)
        else:
            contrib *= 2.0 * np.pi  # already included in frank tamm
        assert np.allclose(contrib, samples["contrib"])


@pytest.mark.parametrize("count", [True, False])
@pytest.mark.parametrize("frankTamm", [True, False])
def test_CherenkovLightSource_bwd(count: bool, frankTamm: bool):
    N = 32 * 256
    dir = np.array([0.36, 0.48, 0.8])
    dist = 200.0 * u.m
    startPos, startT = -dir * 0.5 * dist, -0.5 * dist / u.c
    endPos, endT = dir * 0.5 * dist, 0.5 * dist / u.c

    # build media
    model = PureWaterModel()
    water = model.createMedium(name="water")
    store = MaterialStore([], media=[water])
    waterIdx = store.media["water"]
    # build pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0FFEE)
    photon = theia.light.UniformWavelengthSource()
    source = theia.light.CherenkovLightSource(
        mediumIdx=waterIdx,
        trackStart=startPos,
        trackEnd=endPos,
        startTime=startT,
        endTime=endT,
        usePhotonCount=count,
        applyFrankTamm=frankTamm,
    )
    sampler = BackwardLightSampler(
        N, source, ray, photon, rng=philox, materials=store, medium=waterIdx
    )
    # run
    runPipeline(sampler.collectStages())
    hp.checkCurrentDeviceHealth()
    result = sampler.queue.view(0)

    startTime = result["time"]
    contrib = result["contrib"]
    mask = contrib != 0.0
    assert mask.sum() > 0
    # check results
    assert np.isfinite(result["observer"]).all()
    distA = np.sqrt(np.square(result["position"] - startPos).sum(-1))
    distB = np.sqrt(np.square(result["position"] - endPos).sum(-1))
    assert np.allclose((distA + distB)[mask], dist)  # check position is on track
    assert np.all((startTime[mask] >= startT) & (startTime[mask] <= endT))
    l = distA / dist
    t_exp = (1.0 - l) * startT + l * endT
    assert np.allclose(startTime[mask], t_exp[mask], atol=5e-5)
    exp_dir = result["observer"] - result["position"]
    d = np.sqrt(np.square(exp_dir).sum(-1))
    exp_dir /= d[:, None]
    assert np.allclose(result["direction"], exp_dir)
    cos_theta = np.multiply(result["direction"], dir[None, :]).sum(-1)
    lam = result["wavelength"]
    n = model.refractive_index(lam)
    assert np.allclose(cos_theta, 1.0 / n)
    cos_nrm = np.abs(np.multiply(result["direction"], result["normal"]).sum(-1))
    exp_contrib = cos_nrm / d  # jacobian dA -> dw
    if frankTamm:
        exp_contrib *= theia.light.frankTamm(lam, n, usePhotonCount=count)
        exp_contrib /= 2.0 * np.pi  # specific direction
    assert np.allclose(result["contrib"][mask], exp_contrib[mask], rtol=5e-5)


def trackAngularEmission_pdf(x, n, cos_min=-1, cos_max=1, a=0.39, b=2.61, beta=1.0):
    """PDF of p(x) ~ exp(-bx^a)x^(a-1); x = |cos(theta) - cos(chev)|"""
    # shift x by cherenkov angle
    cos_chev = 1.0 / (beta * n)
    x = np.abs(x - cos_chev)

    # normalization constant
    int_lower = 1.0 - np.exp(-b * np.abs(cos_chev - cos_min) ** a)
    int_upper = 1.0 - np.exp(-b * np.abs(cos_chev - cos_max) ** a)
    # flip sign if cos_chev > cos_min/max
    int_lower *= np.sign(cos_min - cos_chev)
    int_upper *= np.sign(cos_max - cos_chev)
    norm = a * b / np.abs(int_upper - int_lower)

    # calculate pdf
    return np.exp(-b * x**a) * x ** (a - 1) * norm


@pytest.mark.parametrize(
    "n,cos_min,cos_max,a,b,beta",
    [
        [1.33, -1.0, 1.0, 0.39, 2.61, 1.0],
        [1.40, -1.0, -0.5, 0.41, 2.39, 0.9],
        [1.20, 0.8, 0.9, 0.44, 2.78, 1.0],
        [1.35, 0.1, 0.9, 0.41, 2.78, 0.7],
        [1.35, -0.1, 0.9, 0.41, 2.78, 1.0],
    ],
)
def test_trackAngularEmission_pdf(
    n: float,
    cos_min: float,
    cos_max: float,
    a: float,
    b: float,
    beta: float,
) -> None:
    # check wether the pdf integrates to 1
    est, err, *_ = integrate.quad(
        trackAngularEmission_pdf, cos_min, cos_max, (n, cos_min, cos_max, a, b, beta)
    )
    assert np.abs(est - 1.0) < 5e-4
    # check pdf(x) >= 0.0 for all x
    x = np.linspace(cos_min, cos_max, 500)
    assert np.all(trackAngularEmission_pdf(x, n, cos_min, cos_max, a, b, beta) >= 0.0)


@pytest.mark.parametrize(
    "n,a,b",
    [[1.33, 0.53, 3.3], [1.2, 0.86, 2.5], [1.45, 1.03, 3.1], [1.35, 1.13, 3.42]],
)
def test_particle_sampleEmissionAngle_full(n: float, a: float, b: float):
    G = 1024 * 1024
    N = 32 * G
    N_BINS = 100

    tensor = hp.FloatTensor(N)
    buffer = hp.FloatBuffer(N)

    class Push(Structure):
        _fields_ = [("n", c_float), ("a", c_float), ("b", c_float)]

    philox = PhiloxRNG(key=0x01ABBA10)

    code = compileShader(
        "lightsource.particle_sampleEmissionAngle.full.test.glsl",
        headers={"rng.glsl": philox.sourceCode},
    )
    program = hp.Program(code)
    program.bindParams(Samples=tensor)
    philox.bindParams(program, 0)
    philox.update(0)

    push = Push(n, a, b)
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), G))
        .Then(hp.retrieveTensor(tensor, buffer))
        .Submit()
        .wait()
    )

    # check result
    samples = buffer.numpy()
    hist, edges = np.histogram(samples, N_BINS, (-1.0, 1.0), density=True)
    pdf = trackAngularEmission_pdf(edges, n, a=a, b=b)
    exp_hist = 0.5 * (pdf[1:] + pdf[:-1])
    # testing the hist is a bit tricky as the pdf diverges at 1/n
    # -> ignore bin with this value and the ones left and right of it
    peak_bin = int((1.0 / n + 1) / (2.0 / N_BINS))
    err = np.abs(hist - exp_hist) / exp_hist
    assert err[: peak_bin - 1].max() < 0.05
    assert err[peak_bin + 2 :].max() < 0.05


@pytest.mark.parametrize(
    "n,a,b,cos_min,cos_max",
    [
        [1.33, 0.53, 3.3, -1.0, 1.0],
        [1.45, 0.86, 2.8, -0.5, 0.8],
        [1.35, 0.72, 3.1, -0.9, -0.2],
        [1.35, 1.03, 3.1, 0.8, 0.95],
    ],
)
def test_particle_sampleEmissionAngle_range(
    n: float, a: float, b: float, cos_min: float, cos_max: float
) -> None:
    G = 1024 * 1024
    N = 32 * G
    N_BINS = 100

    sample_tensor = hp.FloatTensor(N)
    contrib_tensor = hp.FloatTensor(N)
    sample_buffer = hp.FloatBuffer(N)
    contrib_buffer = hp.FloatBuffer(N)

    class Push(Structure):
        _fields_ = [
            ("n", c_float),
            ("a", c_float),
            ("b", c_float),
            ("cos_min", c_float),
            ("cos_max", c_float),
        ]

    philox = PhiloxRNG(key=0x10500501)

    code = compileShader(
        "lightsource.particle_sampleEmissionAngle.range.test.glsl",
        headers={"rng.glsl": philox.sourceCode},
    )
    program = hp.Program(code)
    program.bindParams(Values=sample_tensor, Contribs=contrib_tensor)
    philox.bindParams(program, 0)
    philox.update(0)

    push = Push(n, a, b, cos_min, cos_max)
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), G))
        .Then(hp.retrieveTensor(sample_tensor, sample_buffer))
        .And(hp.retrieveTensor(contrib_tensor, contrib_buffer))
        .Submit()
        .wait()
    )

    # check result
    samples = sample_buffer.numpy()
    hist, edges = np.histogram(samples, N_BINS, (cos_min, cos_max), density=True)
    pdf = trackAngularEmission_pdf(edges, n, cos_min, cos_max, a, b)
    exp_hist = 0.5 * (pdf[1:] + pdf[:-1])  # trapezoidal rule
    err = np.abs(hist - exp_hist) / exp_hist
    if cos_min <= 1.0 / n <= cos_max:
        # testing the hist is a bit tricky as the pdf diverges at 1/n
        # -> ignore bin with this value and the ones left and right of it
        cos_range = abs(cos_max - cos_min)
        peak_bin = int((1.0 / n - cos_min) / (cos_range / N_BINS))
        assert err[: peak_bin - 1].max() < 0.03
        assert err[peak_bin + 2 :].max() < 0.03
    else:
        assert err.max() < 0.01

    # we do not check contrib here, as we either have to specially deal with the
    # divergence at cos_chev or copy the code from the shader
    # Instead we check it implicitly in the light sources


@pytest.mark.parametrize(
    "count,frankTamm,particle",
    [
        (True, True, False),
        (True, False, True),
        (False, False, False),
        (False, True, False),
    ],
)
def test_MuonTrackLightSource_fwd(count: bool, frankTamm: bool, particle: bool):
    N = 256 * 1024
    lam_range = (300.0, 500.0) * u.nm
    direction, dist = np.array([0.36, 0.48, 0.8]), 100.0 * u.m
    up = np.array([0.8, -0.6, 0.0])  # unit, orthogonal to direction
    startPos, startTime = np.array([1.0, 5.0, -2.0]) * u.m, 0.5 * u.us
    endPos, endTime = startPos + direction * dist, startTime + dist / u.c
    lightParams = {
        "startPosition": startPos,
        "startTime": startTime,
        "endPosition": endPos,
        "endTime": endTime,
        "muonEnergy": 10.0 * u.TeV,
        "applyFrankTamm": frankTamm,
        "usePhotonCount": count,
        "emitParticles": particle,
    }

    # build media
    model = PureWaterModel()
    water = model.createMedium(name="water")
    store = MaterialStore([], media=[water])
    waterIdx = store.media["water"]
    # build pipeline
    ray = UnpolarizedRay(particle=particle)
    philox = PhiloxRNG(key=0xC0FFEE)
    photon = theia.light.UniformWavelengthSource(lambdaRange=lam_range)
    source = theia.light.MuonTrackLightSource(photon, mediumIdx=waterIdx, **lightParams)
    sampler = theia.light.LightSampler(N, source, ray, rng=philox, materials=store)
    # run
    runPipeline(sampler.collectStages())
    hp.checkCurrentDeviceHealth()
    result = sampler.queue.view(0)

    # check all sampled postions are on the track
    distStart = np.sqrt(np.square(startPos - result["position"]).sum(-1))
    distEnd = np.sqrt(np.square(endPos - result["position"]).sum(-1))
    assert np.allclose(distStart + distEnd, dist)
    expTime = startTime + distStart / u.c
    assert np.allclose(result["time"], expTime)
    # check we use the whole track
    assert distStart.min() < 0.05
    assert distEnd.min() < 0.05
    # check direction
    assert np.allclose(np.square(result["direction"]).sum(-1), 1.0)
    cos_theta = np.multiply(result["direction"], direction).sum(-1)
    # since the angular emission pdf is conditional on the refractive index
    # a direct test is a bit complicated (we cannot just create a histogram or
    # use a KS test). For now we just check the range and trust the check of
    # the MC estimate to be enough
    assert cos_theta.max() > 0.999
    assert cos_theta.min() < -0.999
    # check phi is uniform
    cos_phi = np.multiply(result["direction"], up).sum(-1)
    sin_phi = np.multiply(np.cross(result["direction"], up), direction).sum(-1)
    phi = np.arctan2(sin_phi, cos_phi)
    phi = (phi + np.pi) / (2.0 * np.pi)
    ks = stats.kstest(phi[::100], "uniform")
    assert ks.pvalue > 0.05

    if not particle:
        energyScale = source.getParam("_energyScale")
        exp_contrib = np.ones_like(result["contrib"]) * dist * energyScale
        if frankTamm:
            lam = result["wavelength"]
            n = model.refractive_index(lam)
            exp_contrib *= theia.light.frankTamm(lam, n, usePhotonCount=count)
        assert np.allclose(result["contrib"], exp_contrib)


@pytest.mark.parametrize(
    "observer,frankTamm,count",
    [
        [(15.0, 30.0, 60.0) * u.m, True, True],
        [(15.0, 30.0, 60.0) * u.m, True, False],
        [(-10.0, -30.0, 0.0) * u.m, False, True],
        [(-10.0, -30.0, 0.0) * u.m, True, True],
        [(80.0, 80.0, 80.0) * u.m, True, True],
    ],
)
def test_MuonTrackLightSource_bwd(
    observer: tuple[float, float, float], frankTamm: bool, count: bool
):
    N = 256 * 1024
    lam_range = (300.0, 500.0) * u.nm
    startPos, startTime = np.array([1.0, 5.0, -2.0]) * u.m, 0.5 * u.us
    direction, dist = np.array([0.36, 0.48, 0.8]), 100.0 * u.m
    endPos, endTime = startPos + direction * dist, startTime + dist / u.c
    lightParams = {
        "startPosition": startPos,
        "startTime": startTime,
        "endPosition": endPos,
        "endTime": endTime,
        "muonEnergy": 10.0 * u.TeV,
        "applyFrankTamm": frankTamm,
        "usePhotonCount": count,
    }

    # build media
    model = PureWaterModel()
    water = model.createMedium(name="water")
    store = MaterialStore([], media=[water])
    waterIdx = store.media["water"]
    # build pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0FFEE)
    photon = theia.light.UniformWavelengthSource(lambdaRange=lam_range, normalize=False)
    source = theia.light.MuonTrackLightSource(mediumIdx=waterIdx, **lightParams)
    sampler = BackwardLightSampler(
        N,
        source,
        ray,
        photon,
        rng=philox,
        materials=store,
        medium=waterIdx,
        observer=observer,
    )
    # run
    runPipeline(sampler.collectStages())
    result = sampler.queue.view(0)

    # check all sampled postions are on the track
    distStart = np.sqrt(np.square(startPos - result["position"]).sum(-1))
    distEnd = np.sqrt(np.square(endPos - result["position"]).sum(-1))
    assert np.allclose(distStart + distEnd, dist)
    expTime = startTime + distStart / u.c
    assert np.allclose(result["time"], expTime)
    # check we use the whole track
    assert distStart.min() < 0.05
    assert distEnd.min() < 0.05
    # check we aim at observer
    assert np.allclose(result["observer"], observer)
    expDir = observer - result["position"]
    expDir /= np.sqrt(np.square(expDir).sum(-1))[:, None]
    assert np.allclose(result["direction"], expDir)

    # instead of checking the individual sample contributions, we check the
    # corresponding MC estimate, which is the expected number of photons
    # arriving at the observer.
    energyScale = source.getParam("_energyScale")
    a_angular = source.getParam("_a_angular")
    b_angular = source.getParam("_b_angular")

    # integrand
    def f(lam, x):
        # evaluate frank tamm formula
        n = model.refractive_index(lam)
        if frankTamm:
            dN_dx = theia.light.frankTamm(lam, n, usePhotonCount=count)
        else:
            dN_dx = 2.0 * np.pi
        # calculate emission angle
        p = startPos + x * direction
        dir_p = observer - p
        r = np.sqrt(np.square(dir_p).sum(-1))
        dir_p /= r
        cos_theta = np.multiply(dir_p, direction).sum()
        # evaluate emission profile
        ang = trackAngularEmission_pdf(cos_theta, n, a=a_angular, b=b_angular)
        area = 4.0 * np.pi * r**2

        return dN_dx * energyScale * ang / area

    est, err = integrate.dblquad(f, 0, dist, *lam_range, epsrel=1e-4)
    est /= lam_range[1] - lam_range[0]
    assert np.abs(1.0 - result["contrib"].mean() / est) < 8e-3


@pytest.mark.parametrize(
    "count,frankTamm,particle",
    [
        (True, True, False),
        (True, False, True),
        (False, False, False),
        (True, False, False),
        (False, True, False),
    ],
)
@pytest.mark.parametrize("particle_type", ["e+", "e-"])
def test_ParticleCascadeLightSource_fwd(
    frankTamm: bool, count: bool, particle_type: str, particle: bool
):
    N = 256 * 1024
    lam_range = (300.0, 600.0) * u.nm
    startPos, startTime = np.array([1.0, 5.0, -2.0]) * u.m, 0.5 * u.us
    direction = np.array([0.36, 0.48, 0.8])
    up = np.array([0.8, -0.6, 0.0])  # unit, orthogonal to direction
    E_primary = 1.0 * u.TeV
    if particle_type == "e-":
        primary = theia.cascades.ParticleType.E_MINUS
    else:
        primary = theia.cascades.ParticleType.PI_PLUS
    p = theia.cascades.Particle(primary, startPos, direction, E_primary, startTime)
    params = theia.cascades.createParamsFromParticle(p, lightSourceName="")[1]

    # build media
    model = PureWaterModel()
    water = model.createMedium(name="water")
    store = MaterialStore([], media=[water])
    waterIdx = store.media["water"]
    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    philox = PhiloxRNG(key=0xC0FFEE)
    photon = theia.light.UniformWavelengthSource(lambdaRange=lam_range, normalize=False)
    source = theia.light.ParticleCascadeLightSource(
        photon,
        mediumIdx=waterIdx,
        **params,
        applyFrankTamm=frankTamm,
        usePhotonCount=count,
        emitParticles=particle,
    )
    sampler = theia.light.LightSampler(N, source, ray, rng=philox, materials=store)
    # run
    runPipeline(sampler.collectStages())
    hp.checkCurrentDeviceHealth()
    result = sampler.queue.view(0)

    # check all sampled positions are on the track
    dirPos = result["position"] - startPos
    distPos = np.sqrt(np.square(dirPos).sum(-1))
    dirPos = dirPos / distPos[:, None]
    assert np.allclose(dirPos[distPos > 0.0], direction)
    expTime = startTime + distPos / u.c
    assert np.allclose(expTime, result["time"])
    # check we sample along the whole track
    z_max = (params["a_long"] - 1) * params["b_long"]  # mode of gamma dist
    assert distPos.min() < 0.1 * z_max
    assert distPos.max() > 4.0 * z_max  # in theory infinite, but really unlikely
    # check direction
    assert np.allclose(np.square(result["direction"]).sum(-1), 1.0)
    cos_theta = np.multiply(result["direction"], direction).sum(-1)
    # since the angular emission pdf is conditional on the refractive index
    # a direct test is a bit complicated (we cannot just create a histogram or
    # use a KS test). For now we just check the range and trust the check of
    # the MC estimate to be enough
    assert cos_theta.min() < -0.999
    assert cos_theta.max() > 0.999
    # check phi is uniform
    cos_phi = np.multiply(result["direction"], up).sum(-1)
    sin_phi = np.multiply(np.cross(result["direction"], up), direction).sum(-1)
    phi = np.arctan2(sin_phi, cos_phi)
    phi = (phi + np.pi) / (2.0 * np.pi)
    ks = stats.kstest(phi[::100], "uniform")
    assert ks.pvalue > 0.05

    # instead of checking the individual sample contributions we check the MC
    # estimate which is the total number of photons produced
    if not particle:
        est = result["contrib"].mean()
        expEst = params["effectiveLength"]
        if frankTamm:
            lam = np.linspace(*lam_range, 1025)
            dlam = (lam_range[1] - lam_range[0]) / (len(lam) - 1)
            n = model.refractive_index(lam)
            x_ft = theia.light.frankTamm(lam, n, usePhotonCount=count)
            dN_dx = integrate.romb(x_ft, dlam)
            expEst = dN_dx * params["effectiveLength"]
        else:
            # we still have the contrib from the wavelength source
            expEst *= lam_range[1] - lam_range[0]
        assert np.abs(1.0 - est / expEst) < 2e-3


@pytest.mark.parametrize(
    "observer,frankTamm,count",
    [
        [(15.0, 30.0, 60.0) * u.m, True, True],
        [(15.0, 30.0, 60.0) * u.m, True, False],
        [(-10.0, -30.0, 0.0) * u.m, False, True],
        [(80.0, 80.0, 80.0) * u.m, True, True],
    ],
)
@pytest.mark.parametrize("particle_type", ["e-", "pi+"])
def test_ParticleCascadeLightSource_bwd(
    observer: tuple[float, float, float],
    frankTamm: bool,
    count: bool,
    particle_type: str,
):
    N = 256 * 1024
    lam_range = (300.0, 500.0) * u.nm
    startPos, startTime = np.array([1.0, 5.0, -2.0]) * u.m, 0.5 * u.us
    direction = np.array([0.36, 0.48, 0.8])
    up = np.array([0.8, -0.6, 0.0])  # unit, orthogonal to direction
    E_primary = 1.0 * u.TeV
    if particle_type == "e-":
        primary = theia.cascades.ParticleType.E_MINUS
    else:
        primary = theia.cascades.ParticleType.PI_PLUS
    p = theia.cascades.Particle(primary, startPos, direction, E_primary, startTime)
    params = theia.cascades.createParamsFromParticle(p, lightSourceName="")[1]
    a_long, b_long = params["a_long"], params["b_long"]

    # build media
    model = PureWaterModel()
    water = model.createMedium(name="water")
    store = MaterialStore([], media=[water])
    waterIdx = store.media["water"]
    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0FFEE)
    photon = theia.light.UniformWavelengthSource(lambdaRange=lam_range, normalize=False)
    source = theia.light.ParticleCascadeLightSource(
        mediumIdx=waterIdx,
        **params,
        applyFrankTamm=frankTamm,
        usePhotonCount=count,
    )
    sampler = BackwardLightSampler(
        N,
        source,
        ray,
        photon,
        rng=philox,
        materials=store,
        medium=waterIdx,
        observer=observer,
    )
    # run
    runPipeline(sampler.collectStages())
    hp.checkCurrentDeviceHealth()
    result = sampler.queue.view(0)

    # check all sampled positions are on the track
    dirPos = result["position"] - startPos
    distPos = np.sqrt(np.square(dirPos).sum(-1))
    dirPos /= distPos[:, None]
    assert np.allclose(dirPos[distPos > 0.0], direction)
    expTime = startTime + distPos / u.c
    assert np.allclose(expTime, result["time"])
    # check we sample along the whole track
    z_max = (a_long - 1) * b_long  # mode of gamma dist
    assert distPos.min() < 0.1 * z_max
    assert distPos.max() > 4.0 * z_max  # in theory infinite, but really unlikely
    # check we aim at the observer
    assert np.allclose(result["observer"], observer)
    expDir = observer - result["position"]
    expDir /= np.sqrt(np.square(expDir).sum(-1))[:, None]
    assert np.allclose(result["direction"], expDir)

    # instead of checking the individual sample contributions, we check the
    # corresponding MC estimate, which is the expected number of photons
    # arriving at the observer.
    gamma = stats.gamma(a_long)

    # integrand
    def f(lam, x):
        # calculate emission position
        result = gamma.pdf(x / b_long) / b_long
        # calculate emission angle
        p = startPos + x * direction
        dir_p = observer - p
        r = np.sqrt(np.square(dir_p).sum(-1))
        dir_p /= r
        cos_theta = np.multiply(dir_p, direction).sum()
        # evaluate emission profile
        a, b = params["a_angular"], params["b_angular"]
        n = model.refractive_index(lam)
        result *= trackAngularEmission_pdf(cos_theta, n, a=a, b=b)
        result *= params["effectiveLength"]
        result /= 4.0 * np.pi * (r**2)
        # apply frank tamm, if requested
        if frankTamm:
            result *= theia.light.frankTamm(lam, n, usePhotonCount=count)
        else:
            # TODO: Check this. Not totally sure about it
            result *= 2.0 * np.pi
        # done
        return result

    est, err = integrate.dblquad(f, 0, 10.0 * z_max, *lam_range, epsrel=1e-3)
    # we are missing the lambda contrib in our sampler -> get rid of it
    est /= lam_range[1] - lam_range[0]
    assert np.abs(1.0 - result["contrib"].mean() / est) < 0.07
