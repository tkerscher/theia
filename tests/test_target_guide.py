import pytest

import numpy as np
from hephaistos.pipeline import runPipeline

import theia.target
from theia.random import PhiloxRNG
from theia.scene import Transform
from theia.testing import TargetGuideSampler
import theia.units as u


def intersectSphere(center, rad, pos, dir):
    """Intersects sphere at center of radius rad with the ray starting at pos"""
    # see Chapter 7 in "Ray Tracing Gems" by E. Haines et. al.
    f = pos - center
    b2 = np.multiply(f, dir).sum(-1)
    r2 = rad**2
    fd = f - b2[:, None] * dir
    discr = r2 - np.square(fd).sum(-1)
    hit = discr >= 0.0
    c = np.square(f).sum(-1) - r2
    q = -b2 - np.copysign(1.0, b2) * np.sqrt(discr)
    t1, t2 = c / q, q
    # near hit, far hit, hit mask
    return t1, t2, hit


def dist(a, b):
    return np.sqrt(np.square(a - b).sum(-1))


def normalize(v):
    return v / np.sqrt(np.square(v).sum(-1))[:, None]


def test_sphereTargetGuide():
    N = 32 * 256

    # params
    pos = (12.0, -5.0, 3.2)
    radius = 4.0 * u.m

    # create guide and sampler
    guide = theia.target.SphereTargetGuide(position=pos, radius=radius)
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = TargetGuideSampler(N, guide, rng=philox)
    # run
    runPipeline([philox, guide, sampler])

    # check results
    r = sampler.getResults(0)
    pos = np.array(pos)
    d = dist(pos, r["observer"])
    valid = d > radius
    assert np.all(r["sampleProb"][~valid] == 0.0)
    assert np.all((r["sampleProb"] > 0.0) == valid)
    t1, t2, hit = intersectSphere(pos, radius, r["observer"], r["sampleDir"])
    assert np.all(r["sampleDist"][valid] >= t2[valid])
    assert np.allclose(np.square(r["sampleDir"]).sum(-1), 1.0)
    sinOpening = radius / d
    cosMin = 1.0 - np.sqrt(1.0 - sinOpening**2)  # double should have suffice precision
    prob = 1.0 / (2.0 * np.pi * cosMin)
    assert np.allclose(r["sampleProb"][valid], prob[valid])

    assert np.allclose(np.square(r["evalDir"]).sum(-1), 1.0)
    t1, t2, hit = intersectSphere(pos, radius, r["observer"], r["evalDir"])
    valid = valid & (t2 >= 0.0)
    assert np.all(r["evalProb"][~valid] == 0.0)
    assert np.all((r["evalProb"] > 0.0) == valid)
    assert np.all(r["evalDist"][valid] >= t2[valid])
    assert np.allclose(r["evalProb"][valid], prob[valid])


def test_diskTargetGuide():
    N = 32 * 256

    # params
    radius = 12.0 * u.m
    position = (5.0, -2.0, 3.0)
    normal = (-1.0, 2.0, 0.0)
    o2w = Transform.View(position=position, direction=normal)
    w2o = o2w.inverse()

    # create guide and sampler
    guide = theia.target.DiskTargetGuide(
        position=position,
        radius=radius,
        normal=normal,
    )
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = TargetGuideSampler(N, guide, rng=philox)
    # run
    runPipeline([philox, guide, sampler])

    # check results
    r = sampler.getResults(0)
    objObs = w2o.apply(r["observer"])
    valid = objObs[:, 2] > 0.0
    assert np.all((r["sampleProb"] > 0.0) == valid)
    assert np.all((r["sampleProb"] == 0.0) != valid)
    objDir = w2o.applyVec(r["sampleDir"])
    tHit = -objObs[:, 2] / objDir[:, 2]
    assert np.all(tHit[valid] > 0.0)
    assert (r["sampleDist"] - tHit)[valid].min() >= -0.02  # small error is fine
    objHit = objObs + objDir * tHit[:, None]
    assert np.all(np.square(objHit[valid][:, :2]).sum(-1) <= radius**2)
    nrm = np.array(normal)
    nrm /= np.sqrt(np.square(nrm).sum())
    cosNrm = np.abs(np.multiply(r["sampleDir"], nrm[None, :]).sum(-1))
    area = np.pi * radius**2
    prob = 1.0 / area * (tHit**2) / cosNrm
    # rtol maybe a bit large, but needed for larger probs
    # issue due to precision loss? (double vs float)
    assert np.allclose(r["sampleProb"][valid], prob[valid], rtol=2.5e-4)

    objDir = w2o.applyVec(r["evalDir"])
    tHit = -objObs[:, 2] / objDir[:, 2]
    objHit = objObs + objDir * tHit[:, None]
    valid = valid & (tHit > 0.0) & (np.square(objHit)[:, :2].sum(-1) <= radius**2)
    assert valid.sum() > 0
    assert (r["evalDist"] - tHit)[valid].min() >= -1e-5
    assert np.all(r["evalProb"][~valid] == 0.0)
    cosNrm = np.abs(np.multiply(r["evalDir"], nrm[None, :]).sum(-1))
    prob = 1.0 / area * (tHit**2) / cosNrm
    assert np.allclose(r["evalProb"][valid], prob[valid])


def test_flatTargetGuide():
    N = 32 * 256

    # params
    width = 18.0 * u.m
    height = 12.0 * u.m
    position = (5.0, -2.0, 3.0)
    normal = (-1.0, 2.0, 0.0)
    up = (1.0, 1.0, 1.0)
    o2w = Transform.View(position=position, direction=normal, up=up)
    w2o = o2w.inverse()

    # create guide and sampler
    guide = theia.target.FlatTargetGuide(
        width=width,
        height=height,
        position=position,
        normal=normal,
        up=up,
    )
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = TargetGuideSampler(N, guide, rng=philox)
    # run
    runPipeline([philox, guide, sampler])

    # check results
    r = sampler.getResults(0)
    objObs = w2o.apply(r["observer"])
    valid = objObs[:, 2] > 0.0
    assert np.all((r["sampleProb"] > 0.0) == valid)
    assert np.all((r["sampleProb"] == 0.0) != valid)
    objDir = w2o.applyVec(r["sampleDir"])
    tHit = -objObs[:, 2] / objDir[:, 2]
    assert np.all(tHit[valid] > 0.0)
    assert (r["sampleDist"] - tHit)[valid].min() >= -0.002  # small error is fine
    objHit = objObs + objDir * tHit[:, None]
    assert np.all(2.0 * np.abs(objHit).max(0) <= (width, height, 1e-6))
    nrm = np.array(normal)
    nrm /= np.sqrt(np.square(nrm).sum())
    cosNrm = np.abs(np.multiply(r["sampleDir"], nrm[None, :]).sum(-1))
    area = width * height
    prob = 1.0 / area * (tHit**2) / cosNrm
    # rtol maybe a bit large, but needed for larger probs
    # issue due to precision loss? (double vs float)
    assert np.allclose(r["sampleProb"][valid], prob[valid], rtol=2e-4)  # large error?

    objDir = w2o.applyVec(r["evalDir"])
    tHit = -objObs[:, 2] / objDir[:, 2]
    objHit = objObs + objDir * tHit[:, None]
    contained = (2.0 * np.abs(objHit) <= (width, height, 1e-6)).min(-1)
    valid = valid & (tHit > 0.0) & contained
    assert valid.sum() > 0
    assert (r["evalDist"] - tHit)[valid].min() >= -1e-5
    assert np.all(r["evalProb"][~valid] == 0.0)
    cosNrm = np.abs(np.multiply(r["evalDir"], nrm[None, :]).sum(-1))
    prob = 1.0 / area * (tHit**2) / cosNrm
    assert np.allclose(r["evalProb"][valid], prob[valid])
