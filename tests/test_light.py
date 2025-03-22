import pytest

import numpy as np
import scipy.constants as c

import hephaistos as hp
from hephaistos.pipeline import PipelineStage, runPipeline
from theia.util import createPreamble

import theia.light
import theia.material
import theia.random
import theia.units as u

from ctypes import *
from hephaistos.glsl import vec3, vec4
from numpy.lib.recfunctions import structured_to_unstructured

from .common.models import WaterModel


class BackwardLightSampler(PipelineStage):
    """Test program for sampling light in backward mode."""

    class PolarizedItem(Structure):
        _fields_ = [
            ("observer", vec3),
            ("normal", vec3),
            ("wavelength", c_float),
            ("position", vec3),
            ("direction", vec3),
            ("stokes", vec4),
            ("polRef", vec3),
            ("startTime", c_float),
            ("contrib", c_float),
        ]

    class UnpolarizedItem(Structure):
        _fields_ = [
            ("observer", vec3),
            ("normal", vec3),
            ("wavelength", c_float),
            ("position", vec3),
            ("direction", vec3),
            ("startTime", c_float),
            ("contrib", c_float),
        ]

    def __init__(
        self,
        source: theia.light.LightSource,
        capacity: int,
        *,
        rng: theia.random.RNG,
        polarized: bool = False,
        shaderUtil,
    ) -> None:
        super().__init__()

        self._source = source
        self._rng = rng
        self._batch = capacity
        self._groups = -(capacity // -32)

        self._item = self.PolarizedItem if polarized else self.UnpolarizedItem
        self._tensor = hp.ArrayTensor(self._item, capacity)
        self._buffer = [hp.ArrayBuffer(self._item, capacity) for _ in range(2)]

        preamble = createPreamble(POLARIZATION=polarized)
        headers = {
            "rng.glsl": rng.sourceCode,
            "light.glsl": source.sourceCode,
        }
        self._program = shaderUtil.createTestProgram(
            "lightsource.sample.bwd.glsl", preamble, headers
        )
        self._program.bindParams(ResultBuffer=self._tensor)

    def buffer(self, i):
        return self._buffer[i]

    def get(self, i, name):
        return structured_to_unstructured(self._buffer[i].numpy()[name])

    def run(self, i: int):
        self._bindParams(self._program, i)
        self._source.bindParams(self._program, i)
        self._rng.bindParams(self._program, i)
        return [
            self._program.dispatch(self._groups),
            hp.retrieveTensor(self._tensor, self.buffer(i)),
        ]


@pytest.mark.parametrize("polarized", [True, False])
def test_hostLightSource(rng, polarized: bool):
    N = 32 * 256

    # create light & sampler
    photon = theia.light.HostWavelengthSource(N)
    light = theia.light.HostLightSource(N, polarized=polarized)
    sampler = theia.light.LightSampler(light, photon, N, polarized=polarized)

    # fill input buffer
    rays = light.queue.view(0)
    photons = photon.queue.view(0)
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
    photons["wavelength"] = lam
    photons["contrib"] = rng.random((N,)) + 5.0
    rays["startTime"] = rng.random((N,)) * 50.0 * u.ns
    rays["contrib"] = rng.random((N,)) + 1.0
    if polarized:
        # We dont care about valid values
        rays["stokes"] = rng.random((N, 4))
        rays["polarizationRef"] = rng.random((N, 3))

    # run
    runPipeline([photon, light, sampler])
    lightResult = sampler.lightQueue.view(0)
    phResult = sampler.wavelengthQueue.view(0)

    # check result
    assert lightResult.count == N
    assert np.allclose(lightResult["position"], rays["position"])
    assert np.allclose(lightResult["direction"], rays["direction"])
    assert np.allclose(lightResult["startTime"], rays["startTime"])
    assert np.allclose(lightResult["contrib"], rays["contrib"])
    assert np.allclose(phResult["contrib"], photons["contrib"])
    assert np.allclose(phResult["wavelength"], lam)
    if polarized:
        assert np.allclose(lightResult["stokes"], rays["stokes"])
        assert np.allclose(lightResult["polarizationRef"], rays["polarizationRef"])


def test_HostSamplerMismatch():
    photon = theia.light.ConstWavelengthSource()
    with pytest.raises(RuntimeError):
        light = theia.light.HostLightSource(128, polarized=True)
        sampler = theia.light.LightSampler(light, photon, 128, polarized=False)
    with pytest.raises(RuntimeError):
        light = theia.light.HostLightSource(128, polarized=False)
        sampler = theia.light.LightSampler(light, photon, 128, polarized=True)


def test_constWavelength():
    N = 32 * 256
    lam = 450.0 * u.nm

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(lam)
    light = theia.light.PencilLightSource(timeRange=(0.0, 0.0))
    sampler = theia.light.LightSampler(light, photons, N, rng=philox)
    # run
    runPipeline([photons, light, sampler])
    result = sampler.wavelengthQueue.view(0)

    # check result
    assert np.all(result["contrib"] == 1.0)
    assert np.all(result["wavelength"] == lam)


@pytest.mark.parametrize("normalize", [True, False])
def test_uniformWavelength(normalize: bool):
    N = 32 * 256
    lamRange, dLam = (350.0, 750.0) * u.nm, 400.0 * u.nm
    timeRange, dt = (0.0, 0.0) * u.ns, 0.0 * u.ns
    # contrib = L/p(t,lam); p(t, lam) = 1.0 / (|dLam|*|dt|)
    contrib = 1.0 if normalize else dLam

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.UniformWavelengthSource(
        lambdaRange=lamRange, normalize=normalize
    )
    light = theia.light.PencilLightSource(timeRange=timeRange)
    sampler = theia.light.LightSampler(light, photons, N, rng=philox)
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.wavelengthQueue.view(0)

    # check result
    assert np.all(result["contrib"] == contrib)
    assert np.abs(np.min(result["wavelength"]) - lamRange[0]) < 1.0
    assert np.abs(np.max(result["wavelength"]) - lamRange[1]) < 1.0


def test_functionWavelength():
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
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    photons = theia.light.FunctionWavelengthSource(fn, lambdaRange=lamRange)
    light = theia.light.PencilLightSource(timeRange=(0.0, 0.0))
    sampler = theia.light.LightSampler(light, photons, N, rng=philox)
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.wavelengthQueue.view(0)

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


@pytest.mark.parametrize("polarized", [True, False])
def test_coneLightSource_fwd(polarized: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    direction = (0.8, 0.36, 0.48)
    opening = 0.33
    stokes = (1.0, 0.9, 0.1, -0.5)
    polRefIn = (0.0, 0.48, -0.36)
    budget = 12.0

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(wavelength=100.0 * u.nm)
    light = theia.light.ConeLightSource(
        position=position,
        direction=direction,
        timeRange=(10.0, 10.0) * u.ns,
        cosOpeningAngle=opening,
        budget=budget,
        stokes=stokes,
        polarizationReference=polRefIn,
        polarized=polarized,
    )
    sampler = theia.light.LightSampler(
        light, photons, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.lightQueue.view(0)

    # check result
    assert result.count == N
    assert np.allclose(result["position"], position)
    assert (result["direction"] * direction).sum(-1).min() >= opening
    assert np.abs(np.square(result["direction"]).sum(-1) - 1.0).max() < 1e-5
    assert np.all(result["contrib"] == budget)
    if polarized:
        assert np.allclose(result["stokes"], stokes)
        polRef = result["polarizationRef"]
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-5
        # polRef is steepened to be perpendicular to ray dir
        # assert no rotation along ray direction happened, as this would make
        # the stokes vec wrong
        # no rotation happened if the parallelogram has zero volume
        vol = (np.cross(result["direction"], polRefIn) * polRef).sum(-1)
        assert np.abs(vol).max() < 1e-5


@pytest.mark.parametrize("polarized", [True, False])
def test_coneLightSource_bwd(shaderUtil, polarized: bool):
    N = 32 * 256
    light_pos = (14.0, -2.0, 3.0) * u.m
    light_dir = (0.8, 0.36, 0.48)
    light_opening = 0.33
    light_stokes = (1.0, 0.9, 0.1, -0.5)
    polRefIn = (0.0, 0.48, -0.36)
    budget = 12.0

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    light = theia.light.ConeLightSource(
        position=light_pos,
        direction=light_dir,
        timeRange=(10.0, 10.0) * u.ns,
        cosOpeningAngle=light_opening,
        budget=budget,
        stokes=light_stokes,
        polarizationReference=polRefIn,
        polarized=polarized,
    )
    sampler = BackwardLightSampler(
        light, N, rng=philox, polarized=polarized, shaderUtil=shaderUtil
    )
    # run
    runPipeline([philox, light, sampler])
    result = sampler.buffer(0).numpy()

    light_pos = np.array(light_pos)
    light_dir = np.array(light_dir)
    # retrieve results
    observer = structured_to_unstructured(result["observer"])
    position = structured_to_unstructured(result["position"])
    direction = structured_to_unstructured(result["direction"])
    normal = structured_to_unstructured(result["normal"])
    startTime = result["startTime"]
    contrib = result["contrib"]
    # check result
    assert np.isfinite(observer).all()
    assert np.all(position == light_pos)
    exp_dir = observer - light_pos[None, :]
    d = np.sqrt(np.square(exp_dir).sum(-1))
    exp_dir /= d[:, None]
    assert np.allclose(direction, exp_dir)
    assert np.all(startTime == 10.0 * u.ns)
    cos_nrm = np.abs(np.multiply(normal, direction).sum(-1))
    cos_angle = np.multiply(light_dir[None, :], direction).sum(-1)
    exp_contrib = budget * cos_nrm / ((1.0 - light_opening) * 2.0 * np.pi * d**2)
    exp_contrib = np.where(cos_angle >= light_opening, exp_contrib, 0.0)
    assert np.allclose(contrib, exp_contrib)
    # tests for polarization
    if polarized:
        stokes = structured_to_unstructured(result["stokes"])
        polRef = structured_to_unstructured(result["polRef"])
        assert np.allclose(stokes, light_stokes)
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-5
        # polRef is steepened to be perpendicular to ray dir
        # assert no rotation along ray direction happened, as this would make
        # the stokes vec wrong
        # no rotation happened if the parallelogram has zero volume
        vol = (np.cross(direction, polRefIn) * polRef).sum(-1)
        assert np.abs(vol).max() < 1e-5


@pytest.mark.parametrize("polarized", [True, False])
def test_pencilLightSource(polarized: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    direction = (0.8, 0.36, 0.48)  # unit
    stokes = (1.0, 0.9, 0.1, -0.5)
    polRef = (0.0, 0.48, -0.36)
    budget = 12.0
    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(wavelength=100.0 * u.nm)
    light = theia.light.PencilLightSource(
        position=position,
        direction=direction,
        timeRange=(10.0, 10.0) * u.ns,
        budget=budget,
        stokes=stokes,
        polarizationRef=polRef,
    )
    sampler = theia.light.LightSampler(
        light, photons, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.lightQueue.view(0)

    # check result
    assert result.count == N
    assert np.allclose(result["position"], position)
    assert np.allclose(result["direction"], direction)
    assert np.all(result["contrib"] == budget)
    if polarized:
        assert np.allclose(result["stokes"], stokes)
        assert np.allclose(result["polarizationRef"], polRef)


@pytest.mark.parametrize("polarized", [True, False])
def test_sphericalLightSource_fwd(polarized: bool):
    N = 32 * 256
    position = (14.0, -2.0, 3.0) * u.m
    budget = 12.0

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0110FFC0FFEE)
    photons = theia.light.ConstWavelengthSource(wavelength=100.0 * u.nm)
    light = theia.light.SphericalLightSource(
        position=position,
        timeRange=(10.0, 10.0) * u.ns,
        budget=budget,
    )
    sampler = theia.light.LightSampler(
        light, photons, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photons, light, sampler])
    result = sampler.lightQueue.view(0)

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

    if polarized:
        assert np.allclose(result["stokes"], (1.0, 0.0, 0.0, 0.0))
        polRef = result["polarizationRef"]
        assert np.abs((polRef * result["direction"]).sum(-1)).max() < 1e-5
        assert (np.square(polRef).sum(-1) - 1.0).max() < 1e-5


@pytest.mark.parametrize("polarized", [True, False])
def test_sphericalLightSource_bwd(shaderUtil, polarized: bool):
    N = 32 * 256
    light_pos = (14.0, -2.0, 3.0) * u.m
    budget = 1e6

    # create pipeline
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    light = theia.light.SphericalLightSource(
        position=light_pos,
        timeRange=(10.0, 10.0) * u.ns,
        budget=budget,
    )
    sampler = BackwardLightSampler(
        light, N, rng=philox, polarized=polarized, shaderUtil=shaderUtil
    )
    # run
    runPipeline([philox, light, sampler])
    result = sampler.buffer(0).numpy()

    # retrieve results
    observer = structured_to_unstructured(result["observer"])
    position = structured_to_unstructured(result["position"])
    direction = structured_to_unstructured(result["direction"])
    normal = structured_to_unstructured(result["normal"])
    wavelength = result["wavelength"]
    startTime = result["startTime"]
    contrib = result["contrib"]
    light_pos = np.array(light_pos)
    # check result
    assert np.isfinite(observer).all()
    assert np.all(position == light_pos)
    exp_dir = observer - light_pos[None, :]
    d = np.sqrt(np.square(exp_dir).sum(-1))
    exp_dir /= d[:, None]
    assert np.allclose(direction, exp_dir)
    assert np.all(startTime == 10.0 * u.ns)
    cos_nrm = np.abs(np.multiply(normal, direction).sum(-1))
    exp_contrib = budget * cos_nrm / (4.0 * np.pi * d**2)
    # bit larger error, likely precission issues (float vs double)
    assert np.allclose(contrib, exp_contrib, atol=1e-5)
    # tests for polarization
    if polarized:
        stokes = structured_to_unstructured(result["stokes"])
        polRef = structured_to_unstructured(result["polRef"])
        assert np.allclose(stokes, (1.0, 0.0, 0.0, 0.0))
        assert np.abs((polRef * direction).sum(-1)).max() < 1e-5
        assert (np.square(polRef).sum(-1) - 1.0).max() < 1e-5


@pytest.mark.parametrize("usePhotons", [False, True])
@pytest.mark.parametrize("polarized", [True, False])
def test_cherenkov_fwd(usePhotons: bool, polarized: bool):
    N = 32 * 256
    dir = np.array([0.36, 0.48, 0.8])
    dist = 200.0 * u.m
    startPos, startT = -dir * 0.5 * dist, -0.5 * dist / u.c
    endPos, endT = dir * 0.5 * dist, 0.5 * dist / u.c
    # build media
    model = WaterModel()
    water = model.createMedium()
    store = theia.material.MaterialStore([], media=[water])

    # build pipeline
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    photon = theia.light.UniformWavelengthSource(normalize=False)
    light = theia.light.CherenkovLightSource(
        trackStart=startPos,
        trackEnd=endPos,
        startTime=startT,
        endTime=endT,
        medium=store.media["water"],
        usePhotonCount=usePhotons,
    )
    sampler = theia.light.LightSampler(
        light, photon, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photon, light, sampler])
    samples = sampler.lightQueue.view(0)
    photons = sampler.wavelengthQueue.view(0)

    # check result
    distA = np.sqrt(np.square(samples["position"] - startPos).sum(-1))
    distB = np.sqrt(np.square(samples["position"] - endPos).sum(-1))
    assert np.allclose(distA + distB, dist)  # check position is on track
    t = samples["startTime"].ravel()
    assert np.all((t >= startT) & (t <= endT))
    l = distA / dist
    t_exp = (1.0 - l) * startT + l * endT
    assert np.allclose(t, t_exp)
    cos_theta = np.multiply(samples["direction"], dir[None, :]).sum(-1)
    lam = photons["wavelength"]
    n = model.refractive_index(lam)
    assert np.allclose(cos_theta, 1.0 / n)
    contrib = 1.0 - (1.0 / n**2)
    if usePhotons:
        ft_const = 2.0 * np.pi * c.alpha * 1e9
        contrib *= ft_const / (lam**2)
    else:
        ft_const = np.pi * c.e * c.c**2 * c.mu_0 * 1e18
        contrib *= ft_const / (lam**3)
    lam0, lam1 = photon.lambdaRange
    contrib *= dist  # 1/p(lam) 1/p(pos)
    assert np.allclose(contrib, samples["contrib"].ravel())
    assert np.allclose(photons["contrib"], abs(lam1 - lam0))
    # check polarization
    if polarized:
        polRef_exp = np.cross(dir, samples["direction"])
        polRef_exp /= np.sqrt(np.square(polRef_exp).sum(-1))[:, None]
        polRef = samples["polarizationRef"]
        assert np.allclose(samples["stokes"], (1.0, 1.0, 0.0, 0.0))
        assert np.abs(np.abs((polRef * polRef_exp).sum(-1)) - 1.0).max() < 1e-6
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        assert np.abs((polRef * samples["direction"]).sum(-1)).max() < 1e-6


@pytest.mark.parametrize("usePhotons", [False, True])
@pytest.mark.parametrize("polarized", [True, False])
def test_cherenkov_bwd(shaderUtil, usePhotons: bool, polarized: bool):
    N = 32 * 256
    dir = np.array([0.36, 0.48, 0.8])
    dir = np.array([1.0, 0.0, 0.0])
    dist = 200.0 * u.m
    startPos, startT = -dir * 0.5 * dist, -0.5 * dist / u.c
    endPos, endT = dir * 0.5 * dist, 0.5 * dist / u.c
    # build media
    model = WaterModel()
    water = model.createMedium()
    store = theia.material.MaterialStore([], media=[water])

    # build pipeline
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    light = theia.light.CherenkovLightSource(
        trackStart=startPos,
        trackEnd=endPos,
        startTime=startT,
        endTime=endT,
        medium=store.media["water"],
        usePhotonCount=usePhotons,
    )
    sampler = BackwardLightSampler(
        light, N, rng=philox, polarized=polarized, shaderUtil=shaderUtil
    )
    # run
    runPipeline([philox, light, sampler])
    result = sampler.buffer(0).numpy()

    # retrieve results
    observer = structured_to_unstructured(result["observer"])
    position = structured_to_unstructured(result["position"])
    direction = structured_to_unstructured(result["direction"])
    normal = structured_to_unstructured(result["normal"])
    wavelength = result["wavelength"]
    startTime = result["startTime"]
    contrib = result["contrib"]
    mask = contrib != 0.0
    # check results
    assert np.isfinite(observer).all()
    distA = np.sqrt(np.square(position - startPos).sum(-1))
    distB = np.sqrt(np.square(position - endPos)).sum(-1)
    assert np.allclose((distA + distB)[mask], dist)  # check position is on track
    assert np.all((startTime[mask] >= startT) & (startTime[mask] <= endT))
    l = distA / dist
    t_exp = (1.0 - l) * startT + l * endT
    assert np.allclose(startTime[mask], t_exp[mask], atol=5e-5)
    exp_dir = observer - position
    d = np.sqrt(np.square(exp_dir).sum(-1))
    exp_dir /= d[:, None]
    assert np.allclose(direction, exp_dir)
    cos_theta = np.multiply(direction, dir[None, :]).sum(-1)
    n = model.refractive_index(wavelength)
    assert np.allclose(cos_theta, 1.0 / n)
    exp_contrib = 1.0 - (1.0 / n**2)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    cos_nrm = np.abs(np.multiply(direction, normal).sum(-1))
    exp_contrib *= cos_nrm / (sin_theta * d)
    if usePhotons:
        ft_const = c.alpha * 1e9
        exp_contrib *= ft_const / (wavelength**2)
    else:
        ft_const = 0.5 * c.e * c.c**2 * c.mu_0 * 1e18
        exp_contrib *= ft_const / (wavelength**3)
    assert np.allclose(contrib[mask], exp_contrib[mask])
    # check polarization
    if polarized:
        stokes = structured_to_unstructured(result["stokes"])
        polRef = structured_to_unstructured(result["polRef"])
        assert np.allclose(stokes, (1.0, 1.0, 0.0, 0.0))
        polRef_exp = np.cross(dir, direction)
        polRef_exp /= np.sqrt(np.square(polRef_exp).sum(-1))[:, None]
        assert np.abs(np.abs((polRef * polRef_exp).sum(-1)) - 1.0).max() < 1e-6
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        assert np.abs((polRef * direction).sum(-1)).max() < 1e-6


@pytest.mark.parametrize("usePhotons", [False, True])
@pytest.mark.parametrize("polarized", [True, False])
def test_cherenkovTrack(usePhotons: bool, polarized: bool):
    N = 32 * 256
    vertices = np.array([
        #  x,  y,  z, t
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 20.0],
        [1.0, 1.0, 0.0, 35.0],
        [1.0, 1.0, 1.0, 60.0],
    ])
    trackDir = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    dt = np.array([20.0, 15.0, 25.0])

    # build media
    model = WaterModel()
    water = model.createMedium()
    store = theia.material.MaterialStore([], media=[water])

    # build track
    track = theia.light.ParticleTrack(4)
    track.setVertices(vertices)
    # build light source
    photon = theia.light.UniformWavelengthSource(normalize=False)
    light = theia.light.CherenkovTrackLightSource(
        track,
        medium=store.media["water"],
        usePhotonCount=usePhotons,
    )
    # build pipeline
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    sampler = theia.light.LightSampler(
        light, photon, N, rng=philox, polarized=polarized
    )
    # run pipeline
    runPipeline([philox, photon, light, sampler])
    lam0, lam1 = photon.getParam("lambdaRange")
    samples = sampler.lightQueue.view(0)
    photons = sampler.wavelengthQueue.view(0)

    # check result
    assert np.all((samples["position"] >= 0.0) & (samples["position"] <= 1.0))
    t = samples["startTime"].ravel()
    assert np.all((t >= 0.0) & (t <= 60.0))
    t_exp = np.multiply(samples["position"], dt[None, :]).sum(-1)
    assert np.allclose(t, t_exp)
    segmentId = 2 - (samples["position"] == 0.0).sum(-1)
    cos_theta = np.multiply(samples["direction"], trackDir[segmentId]).sum(-1)
    lam = photons["wavelength"].ravel()
    n = model.refractive_index(lam)
    assert np.allclose(cos_theta, 1.0 / n)
    contrib = None
    if usePhotons:
        ft_const = 2.0 * c.pi * c.alpha * 1e9
        # frank tamm; 3.0 from segment sampling
        contrib = ft_const / (lam**2) * (1.0 - (1.0 / n**2)) * 3.0
    else:
        ft_const = c.pi * c.e * c.c**2 * c.mu_0 * 1e18
        contrib = ft_const / (lam**3) * (1.0 - (1.0 / n**2)) * 3.0
    # contrib has additional factor from wavelength sampling
    assert np.allclose(abs(lam1 - lam0), photons["contrib"].ravel())
    assert np.allclose(contrib, samples["contrib"].ravel())
    if polarized:
        polRef_exp = np.cross(trackDir[segmentId], samples["direction"])
        polRef_exp /= np.sqrt(np.square(polRef_exp).sum(-1))[:, None]
        polRef = samples["polarizationRef"]
        assert np.abs(np.abs((polRef * polRef_exp).sum(-1)) - 1.0).max() < 1e-6
        assert np.abs(np.square(polRef).sum(-1) - 1.0).max() < 1e-6
        assert np.abs((polRef * samples["direction"]).sum(-1)).max() < 1e-6
