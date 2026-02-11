import pytest

import numpy as np
from hephaistos.pipeline import runPipeline

import theia.camera
import theia.units as u
from theia.light import ConstWavelengthSource
from theia.material import Material, MaterialStore, PureWaterModel
from theia.random import PhiloxRNG
from theia.ray import UnpolarizedRay
from theia.scene import MeshStore, Scene, Transform
from theia.surface import DielectricSurface
from theia.testing import CameraDirectSampler


def test_HostCamera(rng):
    N = 32 * 256

    # create camera and sampler
    ray = UnpolarizedRay()
    photons = ConstWavelengthSource()
    camera = theia.camera.HostCamera(N, ray, overrideWavelength=False)
    sampler = theia.camera.CameraSampler(N, camera, photons, ray)

    # fill input buffer with random numbers
    raysIn = camera.queue.view(0)
    for field in raysIn.fields:
        raysIn[field] = rng.random(raysIn[field].shape)
    raysIn["objectId"] = rng.integers(-1, 10, N)
    raysIn["mediumIdx"] = rng.integers(0, 10, N)
    # run
    runPipeline(sampler.collectStages())
    # check result
    raysOut = sampler.queue.view(0)
    for field in raysOut.fields:
        assert np.allclose(raysIn[field], raysOut[field])


def test_PencilCamera():
    N = 32 * 256

    # params
    lam = 450.0 * u.nm
    pos = (12.0, -5.0, 3.2)
    dir = (0.36, 0.48, 0.80)  # unit
    delta = 12.5
    hitPos = (13.0, 4.0, -8.0)
    hitDir = (0.48, 0.6, 0.64)  # unit
    hitNrm = (0.6, 0.64, 0.48)  # unit

    # create camera and sampler
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xABBA)
    photon = ConstWavelengthSource(lam)
    camera = theia.camera.PencilCamera(
        mediumIdx=10,
        objectId=4,
        rayPosition=pos,
        rayDirection=dir,
        timeDelta=delta,
        hitPosition=hitPos,
        hitDirection=hitDir,
        hitNormal=hitNrm,
    )
    sampler = theia.camera.CameraSampler(N, camera, photon, ray, rng=philox)
    # run
    runPipeline(sampler.collectStages())

    # check result
    rays = sampler.queue.view(0)
    assert np.allclose(rays["position"], pos)
    assert np.allclose(rays["direction"], dir)
    assert np.allclose(rays["contrib"], 1.0)
    assert np.all(rays["mediumIdx"] == 10)
    assert np.allclose(rays["wavelength"], lam)
    assert np.allclose(rays["time"], delta)
    assert np.allclose(rays["hitPosition"], hitPos)
    assert np.allclose(rays["hitDirection"], hitDir)
    assert np.allclose(rays["hitNormal"], hitNrm)
    assert np.all(rays["objectId"] == 4)


def test_FlatCamera():
    N = 32 * 1024

    # params
    lam = 450.0 * u.nm
    width = 80.0 * u.cm
    length = 60.0 * u.cm
    dx, dy, dz = -2.0, -5.0, 10.0
    trafo = Transform.TRS(rotate=(1.0, 1.0, 0.0, 30.0), translate=(dx, dy, dz))
    camPos = (dx, dy, dz)
    camDir = tuple(trafo.applyVec((0.0, 0.0, 1.0)))
    camUp = tuple(trafo.applyVec((0.0, 1.0, 0.0)))

    # create boundary box from transformed camera corners
    corners = np.vstack(
        [
            trafo.apply((width / 2, length / 2, 0.0)),
            trafo.apply((-width / 2, length / 2, 0.0)),
            trafo.apply((width / 2, -length / 2, 0.0)),
            trafo.apply((-width / 2, -length / 2, 0.0)),
        ]
    )
    upperCorner = corners.max(0)
    lowerCorner = corners.min(0)

    # create camera and sampler
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xABBA)
    photon = ConstWavelengthSource(lam)
    camera = theia.camera.FlatCamera(
        width=width,
        length=length,
        position=camPos,
        direction=camDir,
        up=camUp,
        mediumIdx=10,
        objectId=4,
    )
    sampler = theia.camera.CameraSampler(
        N,
        camera,
        photon,
        ray,
        rng=philox,
    )
    # run
    runPipeline(sampler.collectStages())

    # check result
    rays = sampler.queue.view(0)
    assert np.all(rays["mediumIdx"] == 10)
    assert np.allclose(rays["wavelength"], lam)
    assert np.abs(rays["position"].max(0) - upperCorner).max() < 5e-3
    assert np.abs(rays["position"].min(0) - lowerCorner).max() < 5e-3
    assert np.abs(rays["hitPosition"].min(0) + (width / 2, length / 2, 0)).max() < 5e-5
    assert np.abs(rays["hitPosition"].max(0) - (width / 2, length / 2, 0)).max() < 2e-4
    assert np.all(np.abs(rays["hitPosition"].mean(0)) <= (5e-3, 5e-3, 0.0))
    assert np.abs(trafo.apply(rays["hitPosition"]) - rays["position"]).max() < 1e-6
    assert np.abs(trafo.applyVec(rays["hitDirection"]) + rays["direction"]).max() < 5e-7
    cos_normal = np.abs((rays["hitDirection"] * rays["hitNormal"]).sum(-1))
    assert np.allclose(rays["contrib"], width * length * 2.0 * np.pi * cos_normal)
    assert np.allclose(rays["time"], 0.0)
    assert np.allclose(np.square(rays["hitDirection"]).sum(-1), 1.0)
    assert rays["hitDirection"][:, 2].max() <= 0.0
    assert np.allclose(rays["hitNormal"], (0.0, 0.0, 1.0))
    assert np.all(rays["objectId"] == 4)


def test_FlatCamera_direct():
    N = 32 * 1024

    # params
    width = 80.0 * u.cm
    length = 60.0 * u.cm
    dx, dy, dz = -2.0, -5.0, 10.0
    trafo = Transform.TRS(rotate=(1.0, 1.0, 0.0, 30.0), translate=(dx, dy, dz))
    invTrafo = trafo.inverse()
    normal = (0.0, 0.0, 1.0)
    sampleNrm = trafo.applyVec(normal)  # actually inv(trafo)^T, but here is the same
    camPos = (dx, dy, dz)
    camDir = tuple(trafo.applyVec((0.0, 0.0, 1.0)))
    camUp = tuple(trafo.applyVec((0.0, 1.0, 0.0)))

    # create boundary box from transformed camera corners
    corners = np.vstack(
        [
            trafo.apply((width / 2, length / 2, 0.0)),
            trafo.apply((-width / 2, length / 2, 0.0)),
            trafo.apply((width / 2, -length / 2, 0.0)),
            trafo.apply((-width / 2, -length / 2, 0.0)),
        ]
    )
    upperCorner = corners.max(0)
    lowerCorner = corners.min(0)

    # create camera and sampler
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource()
    camera = theia.camera.FlatCamera(
        width=width,
        length=length,
        position=camPos,
        direction=camDir,
        up=camUp,
        mediumIdx=10,
        objectId=4,
    )
    sampler = CameraDirectSampler(N, ray, camera, photons, rng=philox)
    # run pipeline
    runPipeline(sampler.collectStages())

    # check result
    r = sampler.queue.view(0)
    assert np.abs(r["samplePos"].max(0) - upperCorner).max() < 0.005
    assert np.abs(r["samplePos"].min(0) - lowerCorner).max() < 0.005
    assert np.abs(r["hitPosition"].min(0) + (width / 2, length / 2, 0)).max() < 2e-4
    assert np.abs(r["hitPosition"].max(0) - (width / 2, length / 2, 0)).max() < 6e-5
    assert np.all(np.abs(r["hitPosition"].mean(0)) <= (5e-3, 5e-3, 1e-7))
    assert np.abs(trafo.apply(r["hitPosition"]) - r["position"]).max() < 1e-6
    assert np.abs(trafo.applyVec(r["hitDirection"]) + r["direction"]).max() < 1e-6
    assert np.all(r["position"] == r["samplePos"])
    assert np.allclose(r["direction"], -r["lightDir"])
    assert np.allclose(r["sampleNrm"], sampleNrm)
    assert np.all(r["hitNormal"] == normal)
    assert np.allclose(r["direction"], -trafo.applyVec(r["hitDirection"]), atol=1e-7)
    assert np.allclose(r["time"], 0.0)
    assert np.allclose(r["sampleContrib"], width * length)
    cos_theta = np.multiply(r["direction"], r["sampleNrm"]).sum(-1)
    contrib = cos_theta * width * length * (cos_theta > 0.0).astype(np.float32)
    assert np.allclose(r["contrib"], contrib, atol=1e-7)
    assert np.all(r["objectId"] == 4)
    assert np.all(r["mediumIdx"] == 10)
    assert np.all(r["sampleMediumIdx"] == 10)


def test_ConeCamera():
    N = 32 * 1024

    # params
    lam = 540.0 * u.nm
    pos = (-8.0, 5.4, 3.0)
    dir = (0.36, 0.48, 0.80)  # unit
    theta = 0.12  # opening angle

    # create camera and sampler
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xABBA)
    photon = ConstWavelengthSource(lam)
    camera = theia.camera.ConeCamera(
        position=pos,
        direction=dir,
        cosOpeningAngle=theta,
        mediumIdx=10,
        objectId=4,
    )
    sampler = theia.camera.CameraSampler(
        N,
        camera,
        photon,
        ray,
        rng=philox,
    )
    # run
    runPipeline(sampler.collectStages())

    # check result
    rays = sampler.queue.view(0)
    assert np.all(rays["mediumIdx"] == 10)
    assert np.allclose(rays["wavelength"], lam)
    assert np.allclose(rays["position"], pos)
    assert np.allclose(np.square(rays["direction"]).sum(-1), 1.0)
    assert np.all(np.multiply(rays["direction"], dir).sum(-1) >= theta)
    assert np.allclose(rays["contrib"], 2.0 * np.pi * (1.0 - theta))
    assert np.allclose(rays["time"], 0.0)
    assert np.allclose(rays["hitPosition"], (0.0, 0.0, 0.0))
    assert np.allclose(np.square(rays["hitDirection"]).sum(-1), 1.0)
    assert np.all(rays["hitDirection"][:, 2] <= -theta)
    assert np.allclose(rays["hitNormal"], (0.0, 0.0, 1.0))
    assert np.all(rays["objectId"] == 4)


def test_ConeCamera_direct():
    N = 32 * 1024
    # params
    pos = (-8.0, 5.4, 3.0)
    dir = (0.36, 0.48, 0.80)  # unit
    theta = 0.12  # opening angle

    # create camera and sampler
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource()
    camera = theia.camera.ConeCamera(
        position=pos,
        direction=dir,
        cosOpeningAngle=theta,
        mediumIdx=10,
        objectId=4,
    )
    sampler = CameraDirectSampler(N, ray, camera, photons, rng=philox)
    # run pipeline
    runPipeline(sampler.collectStages())

    # check results
    r = sampler.queue.view(0)
    m = r["contrib"] != 0.0
    assert np.allclose(r["position"], pos)
    assert np.allclose(np.square(r["direction"]).sum(-1), 1.0)
    assert np.all(r["lightDir"] == -r["direction"])
    assert np.all(np.multiply(r["direction"][m], dir).sum(-1) >= theta)
    assert np.allclose(r["sampleNrm"], dir)
    assert np.all(r["contrib"][m] == 1.0)
    assert np.all(r["sampleContrib"] == 1.0)
    assert np.all(r["time"] == 0.0)
    assert np.all(r["position"] == r["samplePos"])
    assert np.allclose(r["hitPosition"], (0.0, 0.0, 0.0))
    assert np.allclose(np.square(r["hitDirection"]).sum(-1), 1.0)
    assert np.all(r["hitDirection"][:, 2][m] <= -theta)
    assert np.allclose(r["hitNormal"], (0.0, 0.0, 1.0))
    assert np.all(r["objectId"] == 4)
    assert np.all(r["mediumIdx"] == 10)
    assert np.all(r["sampleMediumIdx"] == 10)


def test_SphericalCamera():
    N = 32 * 256
    # params
    lam = 450.0 * u.nm
    position = (12.0, 5.0, -7.0)
    radius = 4.0
    t0 = 12.5

    # create camera and sampler
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xABBA)
    photon = ConstWavelengthSource(lam)
    camera = theia.camera.SphereCamera(
        position=position,
        radius=radius,
        timeDelta=t0,
        mediumIdx=10,
        objectId=4,
    )
    sampler = theia.camera.CameraSampler(
        N,
        camera,
        photon,
        ray,
        rng=philox,
    )
    # run
    runPipeline(sampler.collectStages())

    # check result
    rays = sampler.queue.view(0)
    assert np.all(rays["mediumIdx"] == 10)
    assert np.allclose(rays["wavelength"], lam)
    p = np.array(position)
    d = np.sqrt(np.square(rays["position"] - p).sum(-1))
    assert np.allclose(d, radius)
    assert np.abs(rays["hitPosition"].mean(0)).max() < 0.01
    assert np.abs(rays["hitPosition"].var(0) - 1 / 3).max() < 0.01
    assert np.allclose(rays["position"], rays["hitPosition"] * radius + position)
    assert np.allclose(np.square(rays["direction"]).sum(-1), 1.0)
    assert np.allclose(np.square(rays["hitDirection"]).sum(-1), 1.0)
    assert np.allclose(np.square(rays["hitNormal"]).sum(-1), 1.0)
    assert np.allclose(rays["time"], t0)
    cos_normal = np.abs((rays["hitDirection"] * rays["hitNormal"]).sum(-1))
    contrib = 4 * np.pi * radius**2 * 2 * np.pi * cos_normal
    assert np.abs(rays["contrib"] - contrib).max() < 1e-3
    assert np.all(rays["objectId"] == 4)


def test_SphericalCamera_direct():
    N = 32 * 1024
    # params
    position = (12.0, 5.0, -7.0)
    radius = 4.0
    t0 = 12.5

    # create camera and sampler
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource()
    camera = theia.camera.SphereCamera(
        position=position,
        radius=radius,
        timeDelta=t0,
        mediumIdx=10,
        objectId=4,
    )
    sampler = CameraDirectSampler(N, ray, camera, photons, rng=philox)
    # run pipeline
    runPipeline(sampler.collectStages())

    # check results
    r = sampler.queue.view(0)
    d = np.sqrt(np.square(r["position"] - position).sum(-1))
    assert np.allclose(d, radius)
    assert np.abs(r["hitPosition"].mean(0)).max() < 0.01
    assert np.abs(r["hitPosition"].var(0) - 1 / 3).max() < 0.01
    assert np.allclose(r["position"], r["hitPosition"] * radius + position)
    assert np.all(r["position"] == r["samplePos"])
    assert np.allclose(r["sampleNrm"], r["hitNormal"])
    assert np.allclose(r["sampleContrib"], 4.0 * np.pi * radius**2)
    assert np.allclose(r["direction"], -r["lightDir"])
    assert np.allclose(np.square(r["direction"]).sum(-1), 1.0)
    assert np.allclose(np.square(r["hitDirection"]).sum(-1), 1.0)
    assert np.allclose(np.square(r["hitNormal"]).sum(-1), 1.0)
    assert np.allclose(r["time"], t0)
    cos_normal = -(r["hitDirection"] * r["hitNormal"]).sum(-1)
    mask = cos_normal > 0.0
    contrib = 4 * np.pi * radius**2 * cos_normal * mask.astype(np.float32)
    assert np.abs(r["contrib"] - contrib).max() < 5e-5
    assert np.all(r["objectId"] == 4)
    assert np.all(r["mediumIdx"] == 10)
    assert np.all(r["sampleMediumIdx"] == 10)


def test_PointCamera():
    N = 32 * 256
    # params
    lam = 450.0 * u.nm
    position = (12.0, 5.0, -7.0)
    t0 = 12.5

    # create camera and sampler
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xABBA)
    photon = ConstWavelengthSource(lam)
    camera = theia.camera.PointCamera(
        position=position,
        timeDelta=t0,
        mediumIdx=10,
        objectId=4,
    )
    sampler = theia.camera.CameraSampler(
        N,
        camera,
        photon,
        ray,
        rng=philox,
    )
    # run
    runPipeline(sampler.collectStages())

    # check result
    rays = sampler.queue.view(0)
    assert np.all(rays["mediumIdx"] == 10)
    assert np.allclose(rays["wavelength"], lam)
    assert np.allclose(rays["position"], position)
    assert np.allclose(rays["hitPosition"], 0.0)
    assert np.allclose(np.square(rays["direction"]).sum(-1), 1.0)
    assert np.allclose(rays["direction"], -rays["hitDirection"])
    assert np.allclose(rays["direction"], rays["hitNormal"])
    assert np.allclose(rays["contrib"], 4.0 * np.pi)
    assert np.allclose(rays["time"], t0)
    assert np.all(rays["objectId"] == 4)


@pytest.mark.parametrize("inward", [True, False])
def test_MeshCamera(inward: bool):
    N = 32 * 1024
    t0 = 12.5

    # create materials
    water = PureWaterModel().createMedium()
    surface = DielectricSurface()
    mat = Material("mat", None, water, surface)
    matStore = MaterialStore([mat])
    # create scene
    store = MeshStore({"cube": "assets/cube.ply", "sphere": "assets/sphere.stl"})
    t1 = Transform.TRS(
        scale=(3.5, 2.0, 0.5),
        rotate=(1.0, 1.0, 1.0, 115.0),
        translate=(12.5, -5.0, 10.0),
    )
    t2 = Transform.TRS(
        scale=(0.5, 4.0, 3.0),
        rotate=(0.0, -1.0, 0.5, 1.0),
        translate=(0.5, 10.0, -4.0),
    )
    c1 = store.createInstance("cube", "mat", t1)
    c2 = store.createInstance("cube", "mat", t2)
    scene = Scene([c1, c2], matStore)

    # create camera and sampler
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xC0FFEE)
    photon = ConstWavelengthSource()
    camera = theia.camera.MeshCamera(
        c2,
        mediumIdx=10,
        objectId=4,
        timeDelta=t0,
        inward=inward,
    )
    sampler = theia.camera.CameraSampler(
        N,
        camera,
        photon,
        ray,
        rng=philox,
        materials=matStore,
    )
    # run
    runPipeline(sampler.collectStages())

    # check result
    r = sampler.queue.view(0)
    assert np.abs(np.abs(r["hitPosition"]).max(1) - 1.0).max() < 1e-6
    assert np.allclose(r["hitPosition"].min(0), (-1, -1, -1))
    assert np.allclose(r["hitPosition"].max(0), (1, 1, 1))
    assert np.allclose(np.square(r["hitNormal"]).sum(-1), 1.0)
    assert np.abs(np.abs(r["hitNormal"]).max(1) - 1.0).max() < 1e-5
    hit_cos = np.multiply(r["hitNormal"], r["hitDirection"]).sum(-1)
    assert hit_cos.min() >= -1.0 and hit_cos.min() < -0.999
    assert hit_cos.max() <= 0.0 and hit_cos.max() > -0.001
    # if everything's fine, the dot product of hitPos and hitNrm should always
    # be one
    pos_dot = np.multiply(r["hitNormal"], r["hitPosition"]).sum(-1)
    # depending on inward, we either want pos_dot to be 1.0 or -1.0
    pos_dot -= -1.0 if inward else 1.0
    assert np.abs(pos_dot).max() < 1e-5
    assert np.allclose(r["time"], t0)
    # larger error since we offset the ray position to prevent self intersection
    assert np.abs(t2.apply(r["hitPosition"]) - r["position"]).max() < 3e-4
    expDir = t2.applyVec(r["hitDirection"])
    expDir /= np.sqrt(np.square(expDir).sum(-1))[:, None]  # normalize
    assert np.abs(expDir + r["direction"]).max() < 5e-7
