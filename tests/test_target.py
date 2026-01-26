import pytest

import numpy as np
from hephaistos.pipeline import runPipeline

import theia.target
from theia.light import ConstWavelengthSource, LightSampler, SphericalLightSource
from theia.material import VACUUM_IDX
from theia.random import PhiloxRNG
from theia.ray import UnpolarizedRay
from theia.scene import RectBBox, Transform
from theia.testing import TargetSampler
import theia.units as u


def test_DiskTarget():
    N = 32 * 256

    # params
    radius = 12.0 * u.m
    position = (5.0, -2.0, 3.0)
    normal = (-1.0, 2.0, 0.0)
    up = (1.0, 1.0, 1.0)
    o2w = Transform.View(position=position, direction=normal, up=up)
    w2o = o2w.inverse()

    # create target and sampler
    target = theia.target.DiskTarget(
        position=position,
        radius=radius,
        normal=normal,
        up=up,
    )
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = TargetSampler(N, target, rng=philox)
    # run
    runPipeline(sampler.collectStages())

    # check results
    r = sampler.queue.view(0)
    objObs = w2o.apply(r["observer"])
    objDir = w2o.applyVec(r["direction"])
    nrm = np.zeros((N, 3))
    nrm[:, 2] = np.sign(objObs[:, 2])
    nrm = o2w.applyVec(nrm)
    expSampleValid = np.abs(objObs[:, 2]) > 1e-6  # should be close enough
    assert np.all(r["sampleValid"] == expSampleValid)
    assert np.all(r["sampleError"][expSampleValid] == 0)
    assert np.allclose(r["sampleNrm"][expSampleValid], nrm[expSampleValid])
    objPos = w2o.apply(r["samplePos"])
    assert np.all(np.square(objPos[expSampleValid][:, :2]).sum(-1) <= radius**2)
    assert np.abs(objPos[expSampleValid][:, 2]).max() < 1e-6

    t = -objObs[:, 2] / objDir[:, 2]
    objHit = objObs + objDir * t[:, None]
    hit = (t > 0) & (np.square(objHit[:, :2]).sum(-1) <= radius**2)
    hit = hit & expSampleValid
    wHit = o2w.apply(objHit)
    assert hit.sum() > 0  # otherwise tests make no sense
    assert np.all(r["hitValid"] == hit)
    assert np.allclose(r["hitPos"][hit], wHit[hit], atol=5e-6)
    assert np.all(r["hitError"][hit] == 0)
    assert np.allclose(r["hitNrm"][hit], nrm[hit])

    area = np.pi * radius**2
    assert np.allclose(r["sampleProb"][expSampleValid], 1.0 / area)
    assert np.allclose(r["hitProb"][hit], 1.0 / area)


def test_FlatTarget():
    N = 32 * 256

    # params
    width = 12.0 * u.m
    length = 18.0 * u.m
    position = (5.0, -2.0, 3.0)
    normal = (-1.0, 2.0, 0.0)
    up = (1.0, 1.0, 1.0)
    o2w = Transform.View(position=position, direction=normal, up=up)
    w2o = o2w.inverse()

    # create target and sampler
    target = theia.target.FlatTarget(
        width=width,
        length=length,
        position=position,
        direction=normal,
        up=up,
    )
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = TargetSampler(N, target, rng=philox)
    # run
    runPipeline(sampler.collectStages())

    # check result
    r = sampler.queue.view(0)
    objObs = w2o.apply(r["observer"])
    objDir = w2o.applyVec(r["direction"])
    nrm = np.zeros((N, 3))
    nrm[:, 2] = np.sign(objObs[:, 2])
    nrm = o2w.applyVec(nrm)
    expSampleValid = np.abs(objObs[:, 2]) > 1e-6  # should be close enough
    assert np.all(r["sampleValid"] == expSampleValid)
    assert np.all(r["sampleError"][expSampleValid] == 0)
    assert np.allclose(r["sampleNrm"][expSampleValid], nrm[expSampleValid])
    objPos = w2o.apply(r["samplePos"])
    assert np.all(2.0 * np.abs(objPos) <= (width, length, 1e-6))

    t = -objObs[:, 2] / objDir[:, 2]
    objHit = objObs + objDir * t[:, None]
    hit = (t > 0) & (2.0 * np.abs(objHit[:, :2]) <= (width, length)).min(-1)
    hit = hit & expSampleValid
    wHit = o2w.apply(objHit)
    assert hit.sum() > 0  # otherwise the tests make no sense
    assert np.all(r["hitValid"] == hit)
    assert np.allclose(r["hitPos"][hit], wHit[hit], atol=5e-6)
    assert np.all(r["hitError"][hit] == 0)
    assert np.allclose(r["hitNrm"][hit], nrm[hit])

    area = width * length
    assert np.allclose(r["sampleProb"][expSampleValid], 1.0 / area)
    assert np.allclose(r["hitProb"][hit], 1.0 / area)


def test_InnerSphereTarget():
    N = 32 * 256

    # params
    pos = (8.0, -5.0, 3.2)
    radius = 4.8
    box = RectBBox((3.0, -10.0, -1.0), (13.0, 0.0, 8.0))

    # create target and sampler
    target = theia.target.InnerSphereTarget(position=pos, radius=radius)
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    sampler = TargetSampler(N, target, rng=philox, sampleBox=box)
    # run
    runPipeline(sampler.collectStages())

    # check result
    r = sampler.queue.view(0)
    od = np.sqrt(np.square(r["observer"] - pos).sum(-1))
    assert np.all((od >= radius) == r["occluded"])
    oclMask = r["occluded"] == 0  # ignore occluded samples
    expProb = 1.0 / (4.0 * np.pi * radius**2)
    assert np.allclose(r["sampleProb"][oclMask], expProb)
    assert np.all(r["sampleError"][oclMask] == 0)
    # It is safe for the target to assume the rays is not occluded
    # assert (r["sampleValid"] != 0).sum() == N
    # assert (r["sampleValid"] != 0).sum() == oclMask.sum()
    assert (r["sampleValid"] != 0).sum() > 0
    hitMask = oclMask & (r["hitValid"] != 0)
    assert hitMask.sum() > 0
    assert np.all(r["hitError"][hitMask] == 0)
    p = np.array(pos)
    sd = np.sqrt(np.square(r["samplePos"][oclMask] - p).sum(-1))
    assert np.allclose(sd, radius)
    sn = (r["samplePos"][oclMask] - p) / sd[:, None]
    assert np.allclose(-sn, r["sampleNrm"][oclMask], atol=1e-7)
    hd = np.sqrt(np.square(r["hitPos"][hitMask] - p).sum(-1))
    assert np.allclose(hd, radius)
    hn = (r["hitPos"][hitMask] - p) / hd[:, None]
    assert np.allclose(-hn, r["hitNrm"][hitMask])


def test_SphereTarget():
    N = 32 * 256

    # params
    pos = (12.0, -5.0, 3.2)
    radius = 4.0

    # create target and sampler
    target = theia.target.SphereTarget(position=pos, radius=radius)
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = TargetSampler(N, target, rng=philox)
    # run
    runPipeline(sampler.collectStages())

    # check result
    r = sampler.queue.view(0)
    od = np.sqrt(np.square(r["observer"] - pos).sum(-1))
    assert np.all((od <= radius) == r["occluded"])
    oclMask = r["occluded"] == 0  # ignore occluded samples
    area = 2.0 * np.pi * radius**2
    expProb = 1.0 / area / (1.0 - radius / od)
    assert np.allclose(r["sampleProb"][oclMask], expProb[oclMask])
    assert np.all(r["sampleError"][oclMask] == 0)
    assert (r["sampleValid"] != 0).sum() == N
    # It is safe for the target to assume the rays is not occluded
    # assert (r["sampleValid"] != 0).sum() == oclMask.sum()
    hitMask = oclMask & (r["hitValid"] != 0)
    assert hitMask.sum() > 0
    assert np.all(r["hitError"][hitMask] == 0)
    p = np.array(pos)
    sd = np.sqrt(np.square(r["samplePos"][oclMask] - p).sum(-1))
    assert np.allclose(sd, radius)
    sn = (r["samplePos"][oclMask] - p) / sd[:, None]
    assert np.allclose(sn, r["sampleNrm"][oclMask], atol=3e-7)
    hd = np.sqrt(np.square(r["hitPos"][hitMask] - p).sum(-1))
    assert np.allclose(hd, radius)
    hn = (r["hitPos"][hitMask] - p) / hd[:, None]
    assert np.allclose(hn, r["hitNrm"][hitMask])


def test_PointLightTarget():
    N = 32 * 256
    light_pos = (4.0, -2.0, 3.0) * u.m
    budget = 50.0
    target_pos = (1.0, 3.0, -5.0) * u.m
    t_range = (10.0, 10.0) * u.ns

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = ConstWavelengthSource(wavelength=450.0 * u.nm)
    principal = SphericalLightSource(
        photons,
        mediumIdx=VACUUM_IDX,
        position=light_pos,
        timeRange=t_range,
        budget=budget,
    )
    target = theia.target.PointLightSourceTarget(
        mediumIdx=VACUUM_IDX,
        position=target_pos,
    )
    light = theia.target.TargetLightSource(principal, target)
    sampler = LightSampler(N, light, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    exp_dir = np.subtract(target_pos, light_pos)
    dist = np.sqrt(np.square(exp_dir).sum(-1))
    exp_dir /= dist
    assert np.allclose(exp_dir, result["direction"])
    light_pos, target_pos = np.array(light_pos), np.array(target_pos)
    d = np.sqrt(np.square(light_pos - target_pos).sum(-1))
    exp_contrib = budget / (4.0 * np.pi * d**2)
    assert np.allclose(exp_contrib, result["contrib"])


def test_DiskLightTarget():
    N = 32 * 256
    light_pos = (4.0, -2.0, 3.0) * u.m
    budget = 50.0
    t_range = (10.0, 10.0) * u.ns
    target_pos = (1.0, 3.0, -5.0) * u.m
    target_dir = (1.0, 1.0, 1.0)
    target_up = (0.0, 1.0, 0.0)
    radius = 6.0 * u.m
    o2w = Transform.View(position=target_pos, direction=target_dir, up=target_up)
    w2o = o2w.inverse()

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0x01DBEEF)
    photons = ConstWavelengthSource(wavelength=450.0 * u.nm)
    principal = SphericalLightSource(
        photons,
        mediumIdx=VACUUM_IDX,
        position=light_pos,
        timeRange=t_range,
        budget=budget,
    )
    target = theia.target.DiskLightSourceTarget(
        mediumIdx=VACUUM_IDX,
        position=target_pos,
        radius=radius,
        normal=target_dir,
        up=target_up,
    )
    light = theia.target.TargetLightSource(principal, target)
    sampler = LightSampler(N, light, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    objPos = w2o.apply(np.array(light_pos))
    objDir = w2o.applyVec(result["direction"])
    t = -objPos[2] / objDir[:, 2]
    objHit = objPos + objDir * t[:, None]
    hit = (t > 0) & (np.square(objHit[:, :2]).sum(-1) <= radius**2)
    assert np.all(hit)
    cos_nrm = np.abs(objDir[:, 2])  # normal in obj is z
    exp_contrib = budget * radius**2 * cos_nrm / (4.0 * t**2)
    assert np.allclose(exp_contrib, result["contrib"])


def test_FlatLightTarget():
    N = 32 * 256
    light_pos = (4.0, -2.0, 3.0) * u.m
    budget = 50.0
    t_range = (10.0, 10.0) * u.ns
    target_pos = (1.0, 3.0, -5.0) * u.m
    target_dir = (1.0, 1.0, 1.0)
    target_up = (0.0, 1.0, 0.0)
    width, height = (40.0, 60.0) * u.cm
    o2w = Transform.View(position=target_pos, direction=target_dir, up=target_up)
    w2o = o2w.inverse()

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = ConstWavelengthSource(wavelength=450.0 * u.nm)
    principal = SphericalLightSource(
        photons,
        mediumIdx=VACUUM_IDX,
        position=light_pos,
        timeRange=t_range,
        budget=budget,
    )
    target = theia.target.FlatLightSourceTarget(
        mediumIdx=VACUUM_IDX,
        width=width,
        height=height,
        position=target_pos,
        normal=target_dir,
        up=target_up,
    )
    light = theia.target.TargetLightSource(principal, target)
    sampler = LightSampler(N, light, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    objDir = w2o.applyVec(result["direction"])
    objPos = w2o.apply(np.array(light_pos))
    t = -objPos[2] / objDir[:, 2]
    objHit = objPos + objDir * t[:, None]
    hit = (t > 0) & (2.0 * np.abs(objHit[:, :2]) <= (width, height)).min(-1)
    assert np.all(hit)
    cos_nrm = np.abs(objDir[:, 2])  # normal in obj is z
    exp_contrib = width * height * budget * cos_nrm / (4.0 * np.pi * t**2)
    assert np.allclose(exp_contrib, result["contrib"])
