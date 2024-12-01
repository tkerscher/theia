import pytest

import hephaistos as hp
import numpy as np
from hephaistos.pipeline import PipelineStage, runPipeline

import theia.random
import theia.target
from theia.random import RNG, PhiloxRNG
from theia.scene import RectBBox, Transform
import theia.units as u

from ctypes import Structure, c_float
from hephaistos.glsl import vec3
from numpy.lib.recfunctions import structured_to_unstructured


class TargetGuideSampler(PipelineStage):

    class Item(Structure):
        _fields_ = [
            ("observer", vec3),
            ("sampleDir", vec3),
            ("sampleDist", c_float),
            ("sampleProb", c_float),
            ("evalDir", vec3),
            ("evalDist", c_float),
            ("evalProb", c_float),
        ]

    class Push(Structure):
        _fields_ = [
            ("dimMin", vec3),
            ("dimMax", vec3),
        ]

    def __init__(
        self,
        guide: theia.target.TargetGuide,
        capacity: int,
        *,
        rng: RNG,
        sampleBox: RectBBox = RectBBox((-20.0,) * 3, (20.0,) * 3),
        shaderUtil,
    ) -> None:
        super().__init__()

        self._guide = guide
        self._rng = rng
        self._batch = capacity
        self._groups = -(capacity // -32)

        self._push = self.Push(
            dimMin=sampleBox.lowerCorner,
            dimMax=sampleBox.upperCorner,
        )

        self._tensor = hp.ArrayTensor(self.Item, capacity)
        self._buffer = [hp.ArrayBuffer(self.Item, capacity) for _ in range(2)]

        headers = {
            "rng.glsl": rng.sourceCode,
            "target_guide.glsl": guide.sourceCode,
        }
        self._program = shaderUtil.createTestProgram(
            "target_guide.sample.glsl", headers=headers
        )
        self._program.bindParams(ResultBuffer=self._tensor)

    def buffer(self, i):
        return self._buffer[i]

    def getResults(self, i):
        results = {}
        data = self.buffer(i).numpy()
        for name, T in self.Item._fields_:
            if T is c_float:
                results[name] = data[name]
            else:
                results[name] = structured_to_unstructured(data[name])
        return results

    def run(self, i: int):
        self._bindParams(self._program, i)
        self._guide.bindParams(self._program, i)
        self._rng.bindParams(self._program, i)
        return [
            self._program.dispatchPush(bytes(self._push), self._groups),
            hp.retrieveTensor(self._tensor, self._buffer[i]),
        ]


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


def test_sphereTargetGuide(shaderUtil):
    N = 32 * 256

    # params
    pos = (12.0, -5.0, 3.2)
    radius = 4.0 * u.m

    # create guide and sampler
    guide = theia.target.SphereTargetGuide(position=pos, radius=radius)
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    sampler = TargetGuideSampler(guide, N, rng=philox, shaderUtil=shaderUtil)
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


def test_diskTargetGuide(shaderUtil):
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
    sampler = TargetGuideSampler(guide, N, rng=philox, shaderUtil=shaderUtil)
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


def test_flatTargetGuide(shaderUtil):
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
    sampler = TargetGuideSampler(guide, N, rng=philox, shaderUtil=shaderUtil)
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
