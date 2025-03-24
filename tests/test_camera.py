import pytest

import numpy as np
from hephaistos.pipeline import runPipeline

import theia.camera
from theia.light import ConstWavelengthSource, HostWavelengthSource
from theia.material import Material, MaterialStore
from theia.random import PhiloxRNG
from theia.scene import MeshStore, Scene, Transform
from theia.testing import CameraDirectSampler, WaterTestModel
import theia.units as u


@pytest.mark.parametrize("polarized", [True, False])
def test_cameraRaySource(rng, polarized: bool):
    N = 32 * 256

    # create camera and sampler
    photons = HostWavelengthSource(N)
    camera = theia.camera.HostCamera(N, polarized=polarized)
    sampler = theia.camera.CameraRaySampler(camera, photons, N, polarized=polarized)

    # fill input buffer with random numbers
    raysIn = camera.queue.view(0)
    for field in raysIn.fields:
        raysIn[field] = rng.random(raysIn[field].shape)
    phIn = photons.queue.view(0)
    for field in phIn.fields:
        phIn[field] = rng.random(phIn[field].shape)
    # run
    runPipeline([photons, camera, sampler])
    # check result
    raysOut = sampler.cameraQueue.view(0)
    for field in raysOut.fields:
        assert np.allclose(raysIn[field], raysOut[field])
    if polarized:
        assert raysIn.item is theia.camera.PolarizedCameraRayItem
    else:
        assert raysIn.item is theia.camera.CameraRayItem
    phOut = sampler.wavelengthQueue.view(0)
    for field in phOut.fields:
        assert np.allclose(phIn[field], phOut[field])


def test_cameraSamplerMismatch():
    photon = ConstWavelengthSource()
    with pytest.raises(RuntimeError):
        camera = theia.camera.HostCamera(128, polarized=True)
        sampler = theia.camera.CameraRaySampler(camera, photon, 128, polarized=False)
    with pytest.raises(RuntimeError):
        camera = theia.camera.HostCamera(128, polarized=False)
        sampler = theia.camera.CameraRaySampler(camera, photon, 128, polarized=True)


@pytest.mark.parametrize("polarized", [True, False])
def test_pencilCamera(polarized: bool):
    N = 32 * 64

    # params
    pos = (12.0, -5.0, 3.2)
    dir = (0.36, 0.48, 0.80)  # unit
    delta = 12.5
    hitPos = (13.0, 4.0, -8.0)
    hitDir = (0.48, 0.6, 0.64)  # unit
    hitNrm = (0.6, 0.64, 0.48)  # unit
    hitRef = (-0.8, -0.36, -0.48)

    # create camera and sampler
    photon = ConstWavelengthSource()
    camera = theia.camera.PencilCamera(
        rayPosition=pos,
        rayDirection=dir,
        timeDelta=delta,
        hitPosition=hitPos,
        hitDirection=hitDir,
        hitNormal=hitNrm,
        hitPolarizationRef=hitRef,
    )
    sampler = theia.camera.CameraRaySampler(camera, photon, N, polarized=polarized)
    # run
    runPipeline([photon, camera, sampler])

    # check result (use allclose since we compare float <-> double)
    rays = sampler.cameraQueue.view(0)
    assert np.allclose(rays["position"], pos)
    assert np.allclose(rays["direction"], dir)
    assert np.allclose(rays["contrib"], 1.0)
    assert np.allclose(rays["timeDelta"], delta)
    assert np.allclose(rays["hitPosition"], hitPos)
    assert np.allclose(rays["hitDirection"], hitDir)
    assert np.allclose(rays["hitNormal"], hitNrm)
    if polarized:
        # polarization ref was automatically generated, so just test its properties
        assert (rays["direction"] * rays["polarizationRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(rays["polarizationRef"]).sum(-1) - 1.0).max() < 1e-6
        assert (rays["hitDirection"] * rays["hitPolRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(rays["hitPolRef"]).sum(-1) - 1.0).max() < 1e-6


@pytest.mark.parametrize("polarized", [True, False])
def test_flatCamera(polarized: bool):
    N = 32 * 1024

    # params
    width = 80.0 * u.cm
    length = 60.0 * u.cm
    dx, dy, dz = -2.0, -5.0, 10.0
    trafo = Transform.TRS(rotate=(1.0, 1.0, 0.0, 30.0), translate=(dx, dy, dz))
    camPos = (dx, dy, dz)
    camDir = trafo.applyVec((0.0, 0.0, 1.0))
    camUp = trafo.applyVec((0.0, 1.0, 0.0))

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
    photon = ConstWavelengthSource()
    camera = theia.camera.FlatCamera(
        width=width,
        length=length,
        position=camPos,
        direction=camDir,
        up=camUp,
    )
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = theia.camera.CameraRaySampler(
        camera, photon, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photon, camera, sampler])

    # check result
    rays = sampler.cameraQueue.view(0)
    assert np.abs(rays["position"].max(0) - upperCorner).max() < 5e-3
    assert np.abs(rays["position"].min(0) - lowerCorner).max() < 5e-3
    assert np.abs(rays["hitPosition"].min(0) + (width / 2, length / 2, 0)).max() < 5e-5
    assert np.abs(rays["hitPosition"].max(0) - (width / 2, length / 2, 0)).max() < 6e-5
    assert np.all(np.abs(rays["hitPosition"].mean(0)) <= (5e-3, 5e-3, 0.0))
    assert np.abs(trafo.apply(rays["hitPosition"]) - rays["position"]).max() < 1e-6
    assert np.abs(trafo.applyVec(rays["hitDirection"]) + rays["direction"]).max() < 5e-7
    cos_normal = np.abs((rays["hitDirection"] * rays["hitNormal"]).sum(-1))
    assert np.allclose(rays["contrib"], width * length * 2.0 * np.pi * cos_normal)
    assert np.allclose(rays["timeDelta"], 0.0)
    assert np.allclose(np.square(rays["hitDirection"]).sum(-1), 1.0)
    assert rays["hitDirection"][:, 2].max() <= 0.0
    assert np.allclose(rays["hitNormal"], (0.0, 0.0, 1.0))
    if polarized:
        # polarization ref was automatically generated, so just test its properties
        assert (rays["direction"] * rays["polarizationRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(rays["polarizationRef"]).sum(-1) - 1.0).max() < 1e-6
        # check if polRef is perpendicular to plane of scattering
        hitRef = trafo.inverse().applyVec(rays["polarizationRef"])
        assert np.allclose(hitRef, rays["hitPolRef"], atol=1e-7)
        inc = np.cross(rays["hitNormal"], rays["hitDirection"])
        inc /= np.sqrt(np.square(inc).sum(-1))[:, None]
        assert np.abs(np.abs((hitRef * inc).sum(-1)) - 1.0).max() < 1e-6


@pytest.mark.parametrize("polarized", [True, False])
def test_flatCamera_direct(polarized: bool):
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
    camDir = trafo.applyVec((0.0, 0.0, 1.0))
    camUp = trafo.applyVec((0.0, 1.0, 0.0))

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
    camera = theia.camera.FlatCamera(
        width=width,
        length=length,
        position=camPos,
        direction=camDir,
        up=camUp,
    )
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = ConstWavelengthSource()
    sampler = CameraDirectSampler(N, camera, photons, rng=philox, polarized=polarized)
    # run pipeline
    runPipeline(sampler.collectStages())
    r = sampler.getResults(0)

    # check result
    assert np.abs(r["samplePos"].max(0) - upperCorner).max() < 5e-3
    assert np.abs(r["samplePos"].min(0) - lowerCorner).max() < 5e-3
    assert np.abs(r["hitPos"].min(0) + (width / 2, length / 2, 0)).max() < 2e-4
    assert np.abs(r["hitPos"].max(0) - (width / 2, length / 2, 0)).max() < 6e-5
    assert np.all(np.abs(r["hitPos"].mean(0)) <= (5e-3, 5e-3, 1e-7))
    assert np.abs(trafo.apply(r["hitPos"]) - r["rayPos"]).max() < 1e-6
    assert np.abs(trafo.applyVec(r["hitDir"]) + r["rayDir"]).max() < 1e-6
    assert np.all(r["rayPos"] == r["samplePos"])
    assert np.allclose(r["rayDir"], -r["lightDir"])
    assert np.allclose(r["sampleNrm"], sampleNrm)
    assert np.all(r["hitNrm"] == normal)
    assert np.allclose(r["rayDir"], -trafo.applyVec(r["hitDir"]), atol=1e-7)
    assert np.allclose(r["rayTimeDelta"], 0.0)
    assert np.allclose(r["sampleContrib"], width * length)
    cos_theta = np.multiply(r["rayDir"], r["sampleNrm"]).sum(-1)
    contrib = cos_theta * width * length * (cos_theta > 0.0).astype(np.float32)
    assert np.allclose(r["rayContrib"], contrib, atol=1e-7)
    if polarized:
        # polarization ref was automatically generated, so just test its properties
        assert (r["rayDir"] * r["rayPolRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(r["rayPolRef"]).sum(-1) - 1.0).max() < 1e-6
        # check if polRef is perpendicular to plane of scattering
        hitRef = invTrafo.applyVec(r["rayPolRef"])
        assert np.allclose(r["hitPolRef"], hitRef, atol=1e-7)
        inc = np.cross(r["hitNrm"], r["hitDir"])
        inc /= np.sqrt(np.square(inc).sum(-1))[:, None]
        assert np.abs(np.abs((hitRef * inc).sum(-1)) - 1.0).max() < 1e-6


@pytest.mark.parametrize("polarized", [True, False])
def test_coneCamera(polarized: bool):
    N = 32 * 1024

    # params
    pos = (-8.0, 5.4, 3.0)
    dir = (0.36, 0.48, 0.80)  # unit
    theta = 0.12  # opening angle

    # create camera and sampler
    photon = ConstWavelengthSource()
    camera = theia.camera.ConeCamera(position=pos, direction=dir, cosOpeningAngle=theta)
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = theia.camera.CameraRaySampler(
        camera, photon, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([photon, philox, camera, sampler])

    # check result
    rays = sampler.cameraQueue.view(0)
    assert np.allclose(rays["position"], pos)
    assert np.allclose(np.square(rays["direction"]).sum(-1), 1.0)
    assert np.all(np.multiply(rays["direction"], dir).sum(-1) >= theta)
    assert np.allclose(rays["contrib"], 2.0 * np.pi * (1.0 - theta))
    assert np.allclose(rays["timeDelta"], 0.0)
    assert np.allclose(rays["hitPosition"], (0.0, 0.0, 0.0))
    assert np.allclose(np.square(rays["hitDirection"]).sum(-1), 1.0)
    assert np.all(rays["hitDirection"][:, 2] <= -theta)
    assert np.allclose(rays["hitNormal"], (0.0, 0.0, 1.0))
    if polarized:
        # polarization ref was automatically generated, so just test its properties
        assert (rays["direction"] * rays["polarizationRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(rays["polarizationRef"]).sum(-1) - 1.0).max() < 1e-6
        # check if polRef is perpendicular to plane of scattering
        inc = np.cross(dir, rays["direction"])
        inc /= np.sqrt(np.square(inc).sum(-1))[:, None]
        polRef = rays["polarizationRef"]
        assert np.abs(np.abs((polRef * inc).sum(-1)) - 1.0).max() < 1e-6


@pytest.mark.parametrize("polarized", [True, False])
def test_coneCamera_direct(polarized: bool):
    N = 32 * 1024
    # params
    pos = (-8.0, 5.4, 3.0)
    dir = (0.36, 0.48, 0.80)  # unit
    theta = 0.12  # opening angle

    # create camera and sampler
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = ConstWavelengthSource()
    camera = theia.camera.ConeCamera(position=pos, direction=dir, cosOpeningAngle=theta)
    sampler = CameraDirectSampler(N, camera, photons, rng=philox, polarized=polarized)
    # run pipeline
    runPipeline(sampler.collectStages())
    r = sampler.getResults(0)

    # check result
    m = r["rayContrib"] != 0.0
    assert np.allclose(r["rayPos"], pos)
    assert np.allclose(np.square(r["rayDir"]).sum(-1), 1.0)
    assert np.all(r["lightDir"] == -r["rayDir"])
    assert np.all(np.multiply(r["rayDir"][m], dir).sum(-1) >= theta)
    assert np.allclose(r["sampleNrm"], dir)
    assert np.all(r["rayContrib"][m] == 1.0)
    assert np.all(r["sampleContrib"] == 1.0)
    assert np.all(r["rayTimeDelta"] == 0.0)
    assert np.all(r["rayPos"] == r["samplePos"])
    assert np.allclose(r["hitPos"], (0.0, 0.0, 0.0))
    assert np.allclose(np.square(r["hitDir"]).sum(-1), 1.0)
    assert np.all(r["hitDir"][:, 2][m] <= -theta)
    assert np.allclose(r["hitNrm"], (0.0, 0.0, 1.0))
    if polarized:
        # polarization ref was automatically generated, so just test its properties
        assert (r["rayDir"] * r["rayPolRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(r["rayPolRef"]).sum(-1) - 1.0).max() < 1e-6
        # check if polRef is perpendicular to plane of scattering
        inc = np.cross(dir, r["rayDir"])
        inc /= np.sqrt(np.square(inc).sum(-1))[:, None]
        polRef = r["rayPolRef"]
        assert np.abs(np.abs((polRef * inc).sum(-1)) - 1.0).max() < 1e-6

        # again with hit
        assert (r["hitDir"] * r["hitPolRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(r["hitPolRef"]).sum(-1) - 1.0).max() < 1e-6
        inc = np.cross((0.0, 0.0, 1.0), r["hitDir"])
        inc /= np.sqrt(np.square(inc).sum(-1))[:, None]
        polRef = r["hitPolRef"]
        assert np.abs(np.abs((polRef * inc).sum(-1)) - 1.0).max() < 1e-6


@pytest.mark.parametrize("polarized", [True, False])
def test_sphericalCamera(polarized: bool):
    N = 32 * 256
    # params
    position = (12.0, 5.0, -7.0)
    radius = 4.0
    t0 = 12.5

    # create camera and sampler
    photon = ConstWavelengthSource()
    camera = theia.camera.SphereCamera(position=position, radius=radius, timeDelta=t0)
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = theia.camera.CameraRaySampler(
        camera, photon, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photon, camera, sampler])

    # check result
    rays = sampler.cameraQueue.view(0)
    p = np.array(position)
    d = np.sqrt(np.square(rays["position"] - p).sum(-1))
    assert np.allclose(d, radius)
    assert np.abs(rays["hitPosition"].mean(0)).max() < 0.01
    assert np.abs(rays["hitPosition"].var(0) - 1 / 3).max() < 0.01
    assert np.allclose(rays["position"], rays["hitPosition"] * radius + position)
    assert np.allclose(np.square(rays["direction"]).sum(-1), 1.0)
    assert np.allclose(np.square(rays["hitDirection"]).sum(-1), 1.0)
    assert np.allclose(np.square(rays["hitNormal"]).sum(-1), 1.0)
    assert np.allclose(rays["timeDelta"], t0)
    cos_normal = np.abs((rays["hitDirection"] * rays["hitNormal"]).sum(-1))
    contrib = 4 * np.pi * radius**2 * 2 * np.pi * cos_normal
    assert np.abs(rays["contrib"] - contrib).max() < 1e-3
    if polarized:
        # polarization ref was automatically generated, so just test its properties
        assert (rays["direction"] * rays["polarizationRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(rays["polarizationRef"]).sum(-1) - 1.0).max() < 1e-6
        assert (rays["hitDirection"] * rays["hitPolRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(rays["hitPolRef"]).sum(-1) - 1.0).max() < 1e-6
        # check if polRef is perpendicular to plane of scattering
        hitRef = rays["hitPolRef"]
        inc = np.cross(rays["hitNormal"], rays["hitDirection"])
        inc /= np.sqrt(np.square(inc).sum(-1))[:, None]
        assert np.abs(np.abs((hitRef * inc).sum(-1)) - 1.0).max() < 1e-6


@pytest.mark.parametrize("polarized", [True, False])
def test_sphericalCamera_direct(polarized: bool):
    N = 32 * 1024
    # params
    position = (12.0, 5.0, -7.0)
    radius = 4.0
    t0 = 12.5

    # create camera and sampler
    philox = PhiloxRNG(key=0xC0FFEE)
    photons = ConstWavelengthSource()
    camera = theia.camera.SphereCamera(position=position, radius=radius, timeDelta=t0)
    sampler = CameraDirectSampler(N, camera, photons, rng=philox, polarized=polarized)
    # run pipeline
    runPipeline(sampler.collectStages())
    r = sampler.getResults(0)

    # check results
    d = np.sqrt(np.square(r["rayPos"] - position).sum(-1))
    assert np.allclose(d, radius)
    assert np.abs(r["hitPos"].mean(0)).max() < 0.01
    assert np.abs(r["hitPos"].var(0) - 1 / 3).max() < 0.01
    assert np.allclose(r["rayPos"], r["hitPos"] * radius + position)
    assert np.all(r["rayPos"] == r["samplePos"])
    assert np.allclose(r["sampleNrm"], r["hitNrm"])
    assert np.allclose(r["sampleContrib"], 4.0 * np.pi * radius**2)
    assert np.allclose(r["rayDir"], -r["lightDir"])
    assert np.allclose(np.square(r["rayDir"]).sum(-1), 1.0)
    assert np.allclose(np.square(r["hitDir"]).sum(-1), 1.0)
    assert np.allclose(np.square(r["hitNrm"]).sum(-1), 1.0)
    assert np.allclose(r["rayTimeDelta"], t0)
    cos_normal = -(r["hitDir"] * r["hitNrm"]).sum(-1)
    mask = cos_normal > 0.0
    contrib = 4 * np.pi * radius**2 * cos_normal * mask.astype(np.float32)
    assert np.abs(r["rayContrib"] - contrib).max() < 5e-5
    if polarized:
        # polarization ref was automatically generated, so just test its properties
        assert (r["rayDir"] * r["rayPolRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(r["rayPolRef"]).sum(-1) - 1.0).max() < 1e-6
        # check if polRef is perpendicular to plane of scattering
        inc = np.cross(r["sampleNrm"], r["rayDir"])
        inc /= np.sqrt(np.square(inc).sum(-1))[:, None]
        polRef = r["rayPolRef"]
        assert np.abs(np.abs((polRef * inc).sum(-1)) - 1.0).max() < 1e-6

        # again with hit
        assert (r["hitDir"] * r["hitPolRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(r["hitPolRef"]).sum(-1) - 1.0).max() < 1e-6
        inc = np.cross(r["hitNrm"], r["hitDir"])
        inc /= np.sqrt(np.square(inc).sum(-1))[:, None]
        polRef = r["hitPolRef"]
        assert np.abs(np.abs((polRef * inc).sum(-1)) - 1.0).max() < 1e-6


@pytest.mark.parametrize("polarized", [True, False])
def test_pointCamera(polarized: bool):
    N = 32 * 256
    # params
    position = (12.0, 5.0, -7.0)
    t0 = 12.5

    # create camera and sampler
    photons = ConstWavelengthSource()
    camera = theia.camera.PointCamera(position=position, timeDelta=t0)
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = theia.camera.CameraRaySampler(
        camera, photons, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photons, camera, sampler])

    # check result
    rays = sampler.cameraQueue.view(0)
    assert np.allclose(rays["position"], position)
    assert np.allclose(rays["hitPosition"], 0.0)
    assert np.allclose(np.square(rays["direction"]).sum(-1), 1.0)
    assert np.allclose(rays["direction"], -rays["hitDirection"])
    assert np.allclose(rays["direction"], rays["hitNormal"])
    assert np.allclose(rays["contrib"], 4.0 * np.pi)
    assert np.allclose(rays["timeDelta"], t0)
    if polarized:
        # polarization ref was automatically generated, so just test its properties
        assert (rays["direction"] * rays["polarizationRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(rays["polarizationRef"]).sum(-1) - 1.0).max() < 1e-6
        assert (rays["hitDirection"] * rays["hitPolRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(rays["hitPolRef"]).sum(-1) - 1.0).max() < 1e-6


@pytest.mark.parametrize(
    "polarized,inward", [(False, False), (False, True), (True, False)]
)
def test_meshCamera(polarized: bool, inward: bool):
    N = 32 * 1024
    t0 = 12.5

    # create materials
    water = WaterTestModel().createMedium()
    mat = Material("mat", None, water)
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
    scene = Scene([c1, c2], matStore.material, medium=matStore.media["water"])

    # create camera and sampler
    photon = ConstWavelengthSource()
    camera = theia.camera.MeshCamera(c2, timeDelta=t0, inward=inward)
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = theia.camera.CameraRaySampler(
        camera, photon, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photon, camera, sampler])

    # check result
    r = sampler.cameraQueue.view(0)
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
    assert np.allclose(r["timeDelta"], t0)
    # larger error since we offset the ray position to prevent self intersection
    assert np.abs(t2.apply(r["hitPosition"]) - r["position"]).max() < 3e-4
    expDir = t2.applyVec(r["hitDirection"])
    expDir /= np.sqrt(np.square(expDir).sum(-1))[:, None]  # normalize
    assert np.abs(expDir + r["direction"]).max() < 5e-7
    if polarized:
        # polarization ref was automatically generated, so just test its properties
        assert (r["direction"] * r["polarizationRef"]).sum(-1).max() < 1e-6
        assert np.abs(np.square(r["polarizationRef"]).sum(-1) - 1.0).max() < 1e-6
        # check if polRef is perpendicular to plane of scattering
        inc = np.cross(r["hitNormal"], r["hitDirection"])
        inc /= np.sqrt(np.square(inc).sum(-1))[:, None]
        assert np.abs(np.abs((r["hitPolRef"] * inc).sum(-1)) - 1.0).max() < 1e-6
