import pytest

import hephaistos as hp
import numpy as np
from hephaistos.pipeline import PipelineStage, runPipeline
from theia.util import createPreamble

import theia.camera
import theia.light
from theia.random import PhiloxRNG, RNG
from theia.scene import Transform
import theia.units as u

from ctypes import *
from hephaistos.glsl import mat4, vec3, vec4
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.typing import NDArray


class CameraDirectSampler(PipelineStage):
    """Test program for sampling cameras in direct mode"""

    class PolarizedItem(Structure):
        _fields_ = [
            ("wavelength", c_float),
            ("lightDir", vec3),
            ("samplePos", vec3),
            ("sampleNrm", vec3),
            ("sampleContrib", c_float),
            ("rayPos", vec3),
            ("rayDir", vec3),
            ("rayPolRef", vec3),
            ("mueller", mat4),
            ("rayContrib", c_float),
            ("rayTimeDelta", c_float),
            ("hitPolRef", vec3),
            ("hitPos", vec3),
            ("hitDir", vec3),
            ("hitNrm", vec3),
        ]

    class UnpolarizedItem(Structure):
        _fields_ = [
            ("wavelength", c_float),
            ("lightDir", vec3),
            ("samplePos", vec3),
            ("sampleNrm", vec3),
            ("sampleContrib", c_float),
            ("rayPos", vec3),
            ("rayDir", vec3),
            ("rayContrib", c_float),
            ("rayTimeDelta", c_float),
            ("hitPos", vec3),
            ("hitDir", vec3),
            ("hitNrm", vec3),
        ]

    def __init__(
        self,
        camera: theia.camera.CameraRaySource,
        capacity: int,
        *,
        rng: RNG,
        polarized: bool,
        shaderUtil,
    ) -> None:
        super().__init__()

        self._camera = camera
        self._rng = rng
        self._batch = capacity
        self._groups = -(capacity // -32)

        self._item = self.PolarizedItem if polarized else self.UnpolarizedItem
        self._tensor = hp.ArrayTensor(self._item, capacity)
        self._buffer = [hp.ArrayBuffer(self._item, capacity) for _ in range(2)]

        preamble = createPreamble(POLARIZATION=polarized)
        headers = {
            "rng.glsl": rng.sourceCode,
            "camera.glsl": camera.sourceCode,
        }
        self._program = shaderUtil.createTestProgram(
            "camera.direct.sample.glsl", preamble, headers
        )
        self._program.bindParams(ResultBuffer=self._tensor)

    def buffer(self, i):
        return self._buffer[i]

    def get(self, i, name) -> NDArray:
        return structured_to_unstructured(self._buffer[i].numpy()[name])

    def getResults(self, i):
        results = {}
        data = self._buffer[i].numpy()
        for name, T in self._item._fields_:
            if T is c_float:
                results[name] = data[name]
            else:
                results[name] = structured_to_unstructured(data[name])
        return results

    def run(self, i: int):
        self._bindParams(self._program, i)
        self._camera.bindParams(self._program, i)
        self._rng.bindParams(self._program, i)
        return [
            self._program.dispatch(self._groups),
            hp.retrieveTensor(self._tensor, self.buffer(i)),
        ]


@pytest.mark.parametrize("polarized", [True, False])
def test_cameraRaySource(rng, polarized: bool):
    N = 32 * 256

    # create camera and sampler
    photons = theia.light.HostWavelengthSource(N)
    camera = theia.camera.HostCameraRaySource(N, polarized=polarized)
    sampler = theia.camera.CameraRaySampler(camera, photons, N, polarized=polarized)

    # fill input buffer with random numbers
    raysIn = camera.view(0)
    for field in raysIn.fields:
        raysIn[field] = rng.random(raysIn[field].shape)
    phIn = photons.view(0)
    for field in phIn.fields:
        phIn[field] = rng.random(phIn[field].shape)
    # run
    runPipeline([photons, camera, sampler])
    # check result
    raysOut = sampler.cameraView(0)
    for field in raysOut.fields:
        assert np.allclose(raysIn[field], raysOut[field])
    if polarized:
        assert raysIn.item is theia.camera.PolarizedCameraRayItem
    else:
        assert raysIn.item is theia.camera.CameraRayItem
    phOut = sampler.wavelengthView(0)
    for field in phOut.fields:
        assert np.allclose(phIn[field], phOut[field])


def test_cameraSamplerMismatch():
    photon = theia.light.ConstWavelengthSource()
    with pytest.raises(RuntimeError):
        camera = theia.camera.HostCameraRaySource(128, polarized=True)
        sampler = theia.camera.CameraRaySampler(camera, photon, 128, polarized=False)
    with pytest.raises(RuntimeError):
        camera = theia.camera.HostCameraRaySource(128, polarized=False)
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
    photon = theia.light.ConstWavelengthSource()
    camera = theia.camera.PencilCameraRaySource(
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
    rays = sampler.cameraView(0)
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
    trafo = Transform().rotate(1.0, 1.0, 0.0, np.pi / 6).translate(dx, dy, dz)
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
    photon = theia.light.ConstWavelengthSource()
    camera = theia.camera.FlatCameraRaySource(
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
    rays = sampler.cameraView(0)
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
def test_flatCamera_direct(shaderUtil, polarized: bool):
    N = 32 * 1024

    # params
    width = 80.0 * u.cm
    length = 60.0 * u.cm
    dx, dy, dz = -2.0, -5.0, 10.0
    trafo = Transform().rotate(1.0, 1.0, 0.0, np.pi / 6).translate(dx, dy, dz)
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
    camera = theia.camera.FlatCameraRaySource(
        width=width,
        length=length,
        position=camPos,
        direction=camDir,
        up=camUp,
    )
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = CameraDirectSampler(
        camera, N, rng=philox, polarized=polarized, shaderUtil=shaderUtil
    )
    # run pipeline
    runPipeline([philox, camera, sampler])
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
    photon = theia.light.ConstWavelengthSource()
    camera = theia.camera.ConeCameraRaySource(
        position=pos, direction=dir, cosOpeningAngle=theta
    )
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = theia.camera.CameraRaySampler(
        camera, photon, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([photon, philox, camera, sampler])

    # check result
    rays = sampler.cameraView(0)
    assert np.allclose(rays["position"], pos)
    assert np.allclose(np.square(rays["direction"]).sum(-1), 1.0)
    assert np.all(np.multiply(rays["direction"], dir).sum(-1) >= (1.0 - theta))
    assert np.allclose(rays["contrib"], 2.0 * np.pi * theta)
    assert np.allclose(rays["timeDelta"], 0.0)
    assert np.allclose(rays["hitPosition"], (0.0, 0.0, 0.0))
    assert np.allclose(np.square(rays["hitDirection"]).sum(-1), 1.0)
    assert np.all(rays["hitDirection"][:, 2] <= -(1.0 - theta))
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
def test_coneCamera_direct(shaderUtil, polarized: bool):
    N = 32 * 1024
    # params
    pos = (-8.0, 5.4, 3.0)
    dir = (0.36, 0.48, 0.80)  # unit
    theta = 0.12  # opening angle

    # create camera and sampler
    philox = PhiloxRNG(key=0xC0FFEE)
    camera = theia.camera.ConeCameraRaySource(
        position=pos, direction=dir, cosOpeningAngle=theta
    )
    sampler = CameraDirectSampler(
        camera, N, rng=philox, polarized=polarized, shaderUtil=shaderUtil
    )
    # run pipeline
    runPipeline([philox, camera, sampler])
    r = sampler.getResults(0)

    # check result
    m = r["rayContrib"] != 0.0
    assert np.allclose(r["rayPos"], pos)
    assert np.allclose(np.square(r["rayDir"]).sum(-1), 1.0)
    assert np.all(r["lightDir"] == -r["rayDir"])
    assert np.all(np.multiply(r["rayDir"][m], dir).sum(-1) >= (1.0 - theta))
    assert np.allclose(r["sampleNrm"], dir)
    assert np.all(r["rayContrib"][m] == 1.0)
    assert np.all(r["sampleContrib"] == 1.0)
    assert np.all(r["rayTimeDelta"] == 0.0)
    assert np.all(r["rayPos"] == r["samplePos"])
    assert np.allclose(r["hitPos"], (0.0, 0.0, 0.0))
    assert np.allclose(np.square(r["hitDir"]).sum(-1), 1.0)
    assert np.all(r["hitDir"][:, 2][m] <= -(1.0 - theta))
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
    photon = theia.light.ConstWavelengthSource()
    camera = theia.camera.SphereCameraRaySource(
        position=position, radius=radius, timeDelta=t0
    )
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = theia.camera.CameraRaySampler(
        camera, photon, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photon, camera, sampler])

    # check result
    rays = sampler.cameraView(0)
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
def test_sphericalCamera_direct(shaderUtil, polarized: bool):
    N = 32 * 1024
    # params
    position = (12.0, 5.0, -7.0)
    radius = 4.0
    t0 = 12.5

    # create camera and sampler
    philox = PhiloxRNG(key=0xC0FFEE)
    camera = theia.camera.SphereCameraRaySource(
        position=position, radius=radius, timeDelta=t0
    )
    sampler = CameraDirectSampler(
        camera, N, rng=philox, polarized=polarized, shaderUtil=shaderUtil
    )
    # run pipeline
    runPipeline([philox, camera, sampler])
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
    photons = theia.light.ConstWavelengthSource()
    camera = theia.camera.PointCameraRaySource(position=position, timeDelta=t0)
    philox = PhiloxRNG(key=0xC0FFEE)
    sampler = theia.camera.CameraRaySampler(
        camera, photons, N, rng=philox, polarized=polarized
    )
    # run
    runPipeline([philox, photons, camera, sampler])

    # check result
    rays = sampler.cameraView(0)
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
