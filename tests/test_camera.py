import numpy as np
from hephaistos.pipeline import runPipeline

import theia.camera
from theia.random import PhiloxRNG
from theia.scene import Transform
import theia.units as u


def test_cameraRaySource(rng):
    N = 32 * 256

    # create camera and sampler
    camera = theia.camera.HostCameraRaySource(N)
    sampler = theia.camera.CameraRaySampler(camera, N)

    # fill input buffer with random numbers
    raysIn = camera.view(0)
    for field in raysIn.fields:
        raysIn[field] = rng.random(raysIn[field].shape)
    # run
    runPipeline([camera, sampler])
    # check result
    raysOut = sampler.view(0)
    for field in raysOut.fields:
        assert np.allclose(raysIn[field], raysOut[field])


def test_pencilCamera():
    N = 32 * 64

    # params
    pos = (12.0, -5.0, 3.2)
    dir = (0.36, 0.48, 0.80)  # unit
    delta = 12.5
    hitPos = (13.0, 4.0, -8.0)
    hitDir = (0.48, 0.6, 0.64)  # unit
    hitNrm = (0.6, 0.64, 0.48)  # unit

    # create camera and sampler
    camera = theia.camera.PencilCameraRaySource(
        rayPosition=pos,
        rayDirection=dir,
        timeDelta=delta,
        hitPosition=hitPos,
        hitDirection=hitDir,
        hitNormal=hitNrm,
    )
    sampler = theia.camera.CameraRaySampler(camera, N)
    # run
    runPipeline([camera, sampler])

    # check result (use allclose since we compare float <-> double)
    rays = sampler.view(0)
    assert np.allclose(rays["position"], pos)
    assert np.allclose(rays["direction"], dir)
    assert np.allclose(rays["contrib"], 1.0)
    assert np.allclose(rays["timeDelta"], delta)
    assert np.allclose(rays["hitPosition"], hitPos)
    assert np.allclose(rays["hitDirection"], hitDir)
    assert np.allclose(rays["hitNormal"], hitNrm)


def test_flatCamera():
    N = 32 * 1024

    # params
    width = 80.0 * u.cm
    length = 60.0 * u.cm
    dx, dy, dz = -2.0, -5.0, 10.0
    trafo = Transform().rotate(1.0, 1.0, 0.0, np.pi / 6).translate(dx, dy, dz)

    # create camera and sampler
    camera = theia.camera.FlatCameraRaySource(
        width=width, length=length, transform=trafo
    )
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = theia.camera.CameraRaySampler(camera, N, rng=philox)
    # run
    runPipeline([philox, camera, sampler])

    # check result
    rays = sampler.view(0)
    assert np.allclose(trafo.apply(rays["hitPosition"]), rays["position"])
    assert np.abs(trafo.applyVec(rays["hitDirection"]) + rays["direction"]).max() < 5e-7
    assert np.allclose(rays["contrib"], width * length * 2.0 * np.pi)
    assert np.allclose(rays["timeDelta"], 0.0)
    assert np.all(np.abs(rays["hitPosition"].mean(0)) <= (5e-3, 5e-3, 0.0))
    assert np.all(rays["hitPosition"].min(0) >= (-width / 2, -length / 2, 0.0))
    assert np.all(rays["hitPosition"].max(0) <= (width / 2, length / 2, 0.0))
    assert np.allclose(np.square(rays["hitDirection"]).sum(-1), 1.0)
    assert rays["hitDirection"][:, 2].max() <= 0.0
    assert np.allclose(rays["hitNormal"], (0.0, 0.0, 1.0))


def test_coneCamera():
    N = 32 * 1024

    # params
    pos = (-8.0, 5.4, 3.0)
    dir = (0.36, 0.48, 0.80)  # unit
    theta = 0.12  # opening angle

    # create camera and sampler
    camera = theia.camera.ConeCameraRaySource(
        position=pos, direction=dir, cosOpeningAngle=theta
    )
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = theia.camera.CameraRaySampler(camera, N, rng=philox)
    # run
    runPipeline([philox, camera, sampler])

    # check result
    rays = sampler.view(0)
    assert np.allclose(rays["position"], pos)
    assert np.allclose(np.square(rays["direction"]).sum(-1), 1.0)
    assert np.all(np.multiply(rays["direction"], dir).sum(-1) >= (1.0 - theta))
    assert np.allclose(rays["contrib"], 2.0 * np.pi * theta)
    assert np.allclose(rays["timeDelta"], 0.0)
    assert np.allclose(rays["hitPosition"], (0.0, 0.0, 0.0))
    assert np.allclose(np.square(rays["hitDirection"]).sum(-1), 1.0)
    assert np.all(rays["hitDirection"][:, 2] <= -(1.0 - theta))
    assert np.allclose(rays["hitNormal"], (0.0, 0.0, 1.0))


def test_lenseCamera():
    N = 32 * 1024

    # params
    width = 80.0 * u.cm
    length = 60.0 * u.cm
    f = 2000.0 * u.cm
    r = 120.0 * u.cm
    dx, dy, dz = -2.0, -5.0, 10.0
    trafo = Transform().rotate(1.0, 1.0, 0.0, np.pi / 6).translate(dx, dy, dz)

    # create camera and sampler
    camera = theia.camera.LenseCameraRaySource(
        width=width, length=length, focalLength=f, lenseRadius=r, transform=trafo
    )
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = theia.camera.CameraRaySampler(camera, N, rng=philox)
    # run
    runPipeline([philox, camera, sampler])

    # check result
    rays = sampler.view(0)
    assert np.allclose(trafo.apply(rays["hitPosition"]), rays["position"])
    assert np.abs(trafo.applyVec(rays["hitDirection"]) + rays["direction"]).max() < 5e-7
    dir = -rays["hitDirection"]
    lensePos = rays["hitPosition"] + dir * f / dir[:, 2][:, None]
    assert np.allclose(lensePos[:, 2], f)
    assert np.square(lensePos[:, :2]).sum(-1).max() <= r**2
    # theta = rays["hitDirection"][:,2]
    dist = np.sqrt(np.square(rays["hitPosition"] - lensePos).sum(-1))
    contrib = width * length * np.pi * r**2 * (dist**3) / f
    assert np.allclose(rays["contrib"], contrib)
    assert np.allclose(rays["timeDelta"], 0.0)
    assert np.all(np.abs(rays["hitPosition"].mean(0)) <= (5e-3, 5e-3, 0.0))
    assert np.all(rays["hitPosition"].min(0) >= (-width / 2, -length / 2, 0.0))
    assert np.all(rays["hitPosition"].max(0) <= (width / 2, length / 2, 0.0))
    assert np.allclose(rays["hitNormal"], (0.0, 0.0, 1.0))
