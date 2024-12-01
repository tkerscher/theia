import pytest

import hephaistos as hp
import numpy as np
from hephaistos.pipeline import PipelineStage, runPipeline

import theia.random
import theia.target
from theia.random import RNG, PhiloxRNG
from theia.scene import RectBBox, Transform
import theia.units as u

from ctypes import Structure, c_float, c_int32, c_uint32
from hephaistos.glsl import vec3
from numpy.lib.recfunctions import structured_to_unstructured


class TargetSampler(PipelineStage):

    class Item(Structure):
        _fields_ = [
            ("observer", vec3),
            ("direction", vec3),
            ("samplePos", vec3),
            ("sampleNrm", vec3),
            ("sampleProb", c_float),
            ("sampleValid", c_uint32),
            ("sampleError", c_int32),
            ("hitPos", vec3),
            ("hitNrm", vec3),
            ("hitProb", c_float),
            ("hitValid", c_uint32),
            ("hitError", c_int32),
            ("occluded", c_uint32),
        ]

    class Push(Structure):
        _fields_ = [
            ("dimMin", vec3),
            ("dimMax", vec3),
        ]

    def __init__(
        self,
        target: theia.target.Target,
        capacity: int,
        *,
        rng: RNG,
        sampleBox: RectBBox = RectBBox((-20.0,) * 3, (20.0,) * 3),
        shaderUtil,
    ) -> None:
        super().__init__()

        self._target = target
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
            "target.glsl": target.sourceCode,
        }
        self._program = shaderUtil.createTestProgram(
            "target.sample.glsl", headers=headers
        )
        self._program.bindParams(ResultBuffer=self._tensor)

    def buffer(self, i):
        return self._buffer[i]

    def getResults(self, i):
        results = {}
        data = self._buffer[i].numpy()
        for name, T in self.Item._fields_:
            if issubclass(T, (c_float, c_int32, c_uint32)):
                results[name] = data[name]
            else:
                results[name] = structured_to_unstructured(data[name])
        return results

    def run(self, i: int):
        self._bindParams(self._program, i)
        self._target.bindParams(self._program, i)
        self._rng.bindParams(self._program, i)
        return [
            self._program.dispatchPush(bytes(self._push), self._groups),
            hp.retrieveTensor(self._tensor, self._buffer[i]),
        ]


def test_diskTarget(shaderUtil):
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
    sampler = TargetSampler(target, N, rng=philox, shaderUtil=shaderUtil)
    # run
    runPipeline([philox, target, sampler])

    # check results
    r = sampler.getResults(0)
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


def test_flatTarget(shaderUtil):
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
    sampler = TargetSampler(target, N, rng=philox, shaderUtil=shaderUtil)
    # run
    runPipeline([philox, target, sampler])

    # check result
    r = sampler.getResults(0)
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


def test_innerSphereTarget(shaderUtil):
    N = 32 * 256

    # params
    pos = (8.0, -5.0, 3.2)
    radius = 4.8
    box = RectBBox((3.0, -10.0, -1.0), (13.0, 0.0, 8.0))

    # create target and sampler
    target = theia.target.InnerSphereTarget(position=pos, radius=radius)
    philox = theia.random.PhiloxRNG(key=0xC0FFEE)
    sampler = TargetSampler(target, N, rng=philox, sampleBox=box, shaderUtil=shaderUtil)
    # run
    runPipeline([philox, target, sampler])

    # check result
    r = sampler.getResults(0)
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


def test_sphereTarget(shaderUtil):
    N = 32 * 256

    # params
    pos = (12.0, -5.0, 3.2)
    radius = 4.0

    # create target and sampler
    target = theia.target.SphereTarget(position=pos, radius=radius)
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = TargetSampler(target, N, rng=philox, shaderUtil=shaderUtil)
    # run
    runPipeline([philox, target, sampler])

    # check result
    r = sampler.getResults(0)
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
    assert np.allclose(sn, r["sampleNrm"][oclMask], atol=1e-7)
    hd = np.sqrt(np.square(r["hitPos"][hitMask] - p).sum(-1))
    assert np.allclose(hd, radius)
    hn = (r["hitPos"][hitMask] - p) / hd[:, None]
    assert np.allclose(hn, r["hitNrm"][hitMask])
