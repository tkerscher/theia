from __future__ import annotations

import numpy as np
import hephaistos as hp
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import IOQueue

from ctypes import Structure, c_float, c_int32, c_uint32
from hephaistos.glsl import buffer_reference, mat3, mat4x3, vec3

from theia.compiler import createPreamble, compileShader, loadShader
from theia.light import WavelengthSource
from theia.material import MaterialStore
from theia.random import RNG
from theia.ray import RayModel
from theia.scene import MeshInstance, Transform
import theia.units as u

from typing import Callable


__all__ = [
    "Camera",
    "ConeCamera",
    "DiskCamera",
    "FlatCamera",
    "HostCamera",
    "MeshCamera",
    "PencilCamera",
    "PointCamera",
    "SphereCamera",
]


def __dir__():
    return __all__


class Camera(SourceCodeMixin):
    """
    Base class for camera producing camera rays and samples used in backward,
    bidirectional and direct tracing.
    """

    name = "Camera"

    def __init__(
        self,
        *,
        nRNGSamples: int,
        nRNGDirect: int = 0,
        supportDirect: bool = False,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self._nRNGSamples = nRNGSamples
        self._nRNGDirect = nRNGDirect
        self._supportDirect = supportDirect

    @property
    def nRNGDirect(self) -> int:
        """Number of random numbers drawn per sample for direct lighting"""
        return self._nRNGDirect

    @property
    def nRNGSamples(self) -> int:
        """Amount of random numbers drawn per sample"""
        return self._nRNGSamples

    @property
    def supportDirect(self) -> bool:
        """Whether sampling for direct lighting is supported"""
        return self._supportDirect


class CameraSampler(PipelineStage):
    """
    Utility class for sampling rays from a given camera and storing the results
    in a queue.

    Parameters
    ----------
    batchSize: int
        Number of samples to draw per run
    camera: Camera
        Camera to sample
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from
    ray: RayModel
        Model describing the structure of rays to sample
    rng: RNG | None, default=None
        The random number generator used for sampling. May be `None` if `camera`
        does not require random numbers.
    materials: MaterialStore | None, default=None
        Optional store containing material and medium properties the camera may
        reference. If `None`, creates an empty store.
    """

    name = "Camera Sampler"

    class SampleParams(Structure):
        _fields_ = [
            ("_queueAdr", buffer_reference),
            ("_queueSize", c_uint32),
            ("_batchSize", c_uint32),
            ("baseCount", c_uint32),
        ]

    def __init__(
        self,
        batchSize: int,
        camera: Camera,
        wavelengthSource: WavelengthSource,
        ray: RayModel,
        *,
        rng: RNG | None = None,
        materials: MaterialStore | None = None,
    ) -> None:
        super().__init__({"SampleParams": CameraSampler.SampleParams})
        # check if we have a rng if needed
        needRNG = camera.nRNGSamples > 0 or wavelengthSource.nRNGSamples > 0
        if needRNG and rng is None:
            raise ValueError("camera requires a rng but none was given")
        # create queue
        if ray.cameraItem is None:
            raise ValueError("Ray model does not define a camera sample")
        self._queue = IOQueue(
            ray.cameraItem, batchSize, mode="retrieve", skipCounter=True
        )
        assert self._queue.tensor is not None
        # create empty material store if needed
        if materials is None:
            materials = MaterialStore([], mediaSlots=ray.requiredMediumProperties)
        # save params
        self._camera = camera
        self._ray = ray
        self._rng = rng
        self._materials = materials
        self._wavelengthSource = wavelengthSource
        self.setParams(
            _queueAdr=self._queue.tensor.address,
            _queueSize=batchSize,
            _batchSize=batchSize,
            baseCount=0,
        )

        # create program
        headers = {
            "rng.glsl": "" if rng is None else rng.sourceCode,
            "ray.glsl": ray.sourceCode,
            "photon.glsl": wavelengthSource.sourceCode,
            "camera.glsl": camera.sourceCode,
            **materials.header,
        }
        code = compileShader("camera/sample.glsl", "", headers)
        self._program = hp.Program(code)
        # bind materials
        materials.bindParams(self._program)

    @property
    def batchSize(self) -> int:
        """Number of samples drawn per run"""
        return self._batchSize

    @property
    def camera(self) -> Camera:
        """Camera being sampled"""
        return self._camera

    @property
    def materials(self) -> MaterialStore:
        """Optional material store in use"""
        return self._materials

    @property
    def queue(self) -> IOQueue:
        """Queue holding the sampled light rays"""
        return self._queue

    @property
    def rayModel(self) -> RayModel:
        """Ray model used for sampling"""
        return self._ray

    @property
    def rng(self) -> RNG | None:
        """The random number generator used for sampling. None, if none used."""
        return self._rng

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source producing wavelength samples"""
        return self._wavelengthSource

    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all stages involved with this sampler in the correct
        order suitable for creating a pipeline.
        """
        stages: list[PipelineStage] = []
        if self.rng is not None:
            stages.append(self.rng)
        stages.extend([self.wavelengthSource, self.camera, self])
        return stages

    def run(self, i: int) -> list[hp.Command]:
        for stage in self.collectStages():
            stage.bindParams(self._program, i)
        groups = -(self.batchSize // -512)
        return [self._program.dispatch(groups), *self.queue.run(i)]


class HostCamera(Camera):
    """
    Camera passing rays from the CPU to the GPU.

    Parameters
    ----------
    capacity: int
        Capacity of the underlying queue feeding the GPU.
    ray: RayModel
        Model describing the structure of rays.
    overrideWavelength: bool, default=True
        Whether to use the wavelength from the queue instead of the one passed
        to the camera when sampling.
    updateFn: (HostCamera, int) -> None | None, default=None
        Optional update function called before the pipeline processes a task.
        `i` is the i-th configuration the update should affect.
        Can be used to stream in new samples on demand.
    """

    name = "Host Camera"

    class CameraParams(Structure):
        _fields_ = [
            ("_queueAdr", buffer_reference),
            ("_queueSize", c_uint32),
        ]

    def __init__(
        self,
        capacity: int,
        ray: RayModel | type[Structure],
        *,
        overrideWavelength: bool = True,
        updateFn: Callable[[HostCamera, int], None] | None = None,
        extra: set[str] = set(),
    ) -> None:
        super().__init__(
            nRNGSamples=0,
            params={"CameraParams": HostCamera.CameraParams},
            extra=extra,
        )

        if isinstance(ray, RayModel):
            if ray.cameraItem is None:
                raise ValueError("ray model does not provide a camera item")
            ray = ray.cameraItem
        self._queue = IOQueue(ray, capacity, mode="update", skipCounter=True)
        assert self._queue.tensor is not None
        # save params
        self._overrideLambda = overrideWavelength
        self._updateFn = updateFn
        self.setParams(_queueAdr=self._queue.tensor.address, _queueSize=capacity)

    @property
    def queue(self) -> IOQueue:
        """Queue holding the samples for the next batch"""
        return self._queue

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(CAMERA_OVERRIDE_WAVELENGTH=self._overrideLambda)
        code = loadShader("camera/host.glsl")
        return preamble + code

    def update(self, i: int) -> None:
        if self._updateFn is not None:
            self._updateFn(self, i)
        super().update(i)

    def run(self, i: int) -> list[hp.Command]:
        return [*self.queue.run(i), *super().run(i)]


class PencilCamera(Camera):
    """
    Sampler outputting a constant camera ray with independent hit
    parameterization.

    Parameters
    ----------
    mediumIdx: int
        Index of the medium the camera is submerged in.
    rayPosition: (float, float, float), default=(0.0, 0.0, 0.0)
        Start point of the camera ray
    rayDirection: (float, float, float), default=(0.0, 0.0, 1.0)
        Direction of the camera ray
    timeDelta: float, default=0.0
        Extra time delay added on the ray
    hitPosition: (float, float, float), default=(0.0,0.0,0.0)
        Hit position in the local frame of the detector
    hitDirection: (float, float, float), default=(0.0,0.0,-1.0)
        Ray direction at the hit position in the local frame of the detector
    hitNormal: (float, float, float), default=(0.0,0.0,1.0)
        Surface normal at the hit position in the local frame of the detector
    objectId: int, default=-1
        Id assigned to hits produced by this camera.

    Stage Parameters
    ----------------
    mediumIdx: int
        Index of the medium the camera is submerged in.
    rayPosition: (float, float, float), default=(0.0, 0.0, 0.0)
        Start point of the camera ray
    rayDirection: (float, float, float), default=(0.0, 0.0, 1.0)
        Direction of the camera ray
    timeDelta: float, default=0.0
        Extra time delay added on the ray
    hitPosition: (float, float, float), default=(0.0,0.0,0.0)
        Hit position in the local frame of the detector
    hitDirection: (float, float, float), default=(0.0,0.0,-1.0)
        Ray direction at the hit position in the local frame of the detector
    hitNormal: (float, float, float), default=(0.0,0.0,1.0)
        Surface normal at the hit position in the local frame of the detector
    objectId: int, default=-1
        Id assigned to hits produced by this camera.
    """

    name = "Pencil Camera"

    class CameraParams(Structure):
        _fields_ = [
            ("rayPosition", vec3),
            ("rayDirection", vec3),
            ("timeDelta", c_float),
            ("mediumIdx", c_uint32),
            ("objectId", c_int32),
            ("hitPosition", vec3),
            ("hitDirection", vec3),
            ("hitNormal", vec3),
        ]

    def __init__(
        self,
        *,
        mediumIdx: int,
        rayPosition: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rayDirection: tuple[float, float, float] = (0.0, 0.0, 1.0),
        timeDelta: float = 0.0,
        hitPosition: tuple[float, float, float] = (0.0, 0.0, 0.0),
        hitDirection: tuple[float, float, float] = (0.0, 0.0, -1.0),
        hitNormal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        objectId: int = -1,
    ) -> None:
        super().__init__(
            nRNGSamples=0,
            supportDirect=False,
            params={"CameraParams": self.CameraParams},
        )
        self.setParams(
            mediumIdx=mediumIdx,
            rayPosition=rayPosition,
            rayDirection=rayDirection,
            timeDelta=timeDelta,
            objectId=objectId,
            hitPosition=hitPosition,
            hitDirection=hitDirection,
            hitNormal=hitNormal,
        )

    @property
    def sourceCode(self) -> str:
        return loadShader("camera/pencil.glsl")


class FlatCamera(Camera):
    """
    Camera simulating a rectangle as detector surface, i.e. samples first a
    point on the rectangle followed by a random direction in the upper
    hemisphere.

    In the local coordinates system of hits the rectangle lies in the xy plane
    centered in the origin with its normal facing in positive z direction. Width
    is its size in the x direction, length in the y direction. The orientation
    in global space are determined by `direction` defining the surface normal
    and a `up` vector defining where the local y axis points to.

    Parameters
    ----------
    mediumIdx: int
        Index of the medium the camera is submerged in.
    width: float, default=1cm
        Width of the detector. Corresponds in local space to the camera
        surface's extension in x direction.
    length: float, default=1cm
        Length of the detector. Corresponds in local space to the camera
        surface's extension in y direction.
    position: (float, float, float), default=(0.0,0.0,0.0)
        Position of the camera in world space.
    direction: (float, float, float), default=(0.0,0.0,1.0)
        Direction the camera faces. Corresponds in local space positive z
        direction.
    up: (float, float, float), default=(0.0,1.0,0.0)
        Direction identifying where 'up' is for the camera. Corresponds in local
        space to the positive y direction.
    objectId: int, default=-1
        Id assigned to hits produced by this camera.

    Stage Parameters
    ----------------
    mediumIdx: int
        Index of the medium the camera is submerged in.
    width: float, default=1cm
        Width of the detector. Corresponds in local space to the camera
        surface's extension in x direction.
    length: float, default=1cm
        Length of the detector. Corresponds in local space to the camera
        surface's extension in y direction.
    position: (float, float, float), default=(0.0,0.0,0.0)
        Position of the camera in world space.
    direction: (float, float, float), default=(0.0,0.0,1.0)
        Direction the camera faces. Corresponds in local space positive z
        direction.
    up: (float, float, float), default=(0.0,1.0,0.0)
        Direction identifying where 'up' is for the camera. Corresponds in local
        space to the positive y direction.
    objectId: int, default=-1
        Id assigned to hits produced by this camera.

    Note
    ----
    `direction` and `up` may not be parallel.
    """

    name = "Flat Camera"

    class CameraParams(Structure):
        _fields_ = [
            ("width", c_float),
            ("length", c_float),
            ("objectId", c_int32),
            ("mediumIdx", c_uint32),
            ("position", vec3),
            ("_view", mat3),
        ]

    def __init__(
        self,
        *,
        mediumIdx: int,
        width: float = 1.0 * u.cm,
        length: float = 1.0 * u.cm,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        up: tuple[float, float, float] = (0.0, 1.0, 0.0),
        objectId: int = -1,
    ) -> None:
        super().__init__(
            nRNGSamples=4,
            nRNGDirect=2,
            supportDirect=True,
            params={"CameraParams": self.CameraParams},
            extra={"direction", "up"},
        )
        # save params
        self.position = position
        self.direction = direction
        self.up = up
        self.setParams(
            width=width,
            length=length,
            position=position,
            mediumIdx=mediumIdx,
            objectId=objectId,
        )

    @property
    def direction(self) -> tuple[float, float, float]:
        """Direction the camera faces"""
        return self._direction

    @direction.setter
    def direction(self, value: tuple[float, float, float]) -> None:
        self._direction = value

    @property
    def up(self) -> tuple[float, float, float]:
        """Direction of the local y-Axis identifying where 'up' us for the camera"""
        return self._up

    @up.setter
    def up(self, value: tuple[float, float, float]) -> None:
        self._up = value

    @property
    def sourceCode(self) -> str:
        return loadShader("camera/flat.glsl")

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)

        # create view matrix
        t = Transform.View(direction=self.getParam("direction"), up=self.getParam("up"))
        self.setParam("_view", t.innerMatrix)  # column major


class DiskCamera(Camera):
    """
    Camera simulating a flat circular disk as detector surface, i.e. samples
    first a random point on the disk followed by a random direction in the upper
    hemisphere.

    In the local coordinates system of hits the disk lies in the xy plane
    centered in the origin with its normal facing in positive z direction.
    The orientation in global space are determined by the `direction` defining
    the surface normal and a `up` vector defining where the local y axis points
    to.

    Parameters
    ----------
    mediumIdx: int
        Index of the medium the camera is submerged in
    radius: float, default=1.0
        Radius of the camera disk
    position: (float, float, float), default=(0.0,0.0,0.0)
        Position of the camera in world space
    direction: (float, float, float), default=(0.0,0.0,1.0)
        Direction the camera faces. Corresponds in local space positive z
        direction.
    up: (float, float, float), default=(0.0,1.0,0.0)
        Direction identifying where 'up' is for the camera. Corresponds in local
        space to the positive y direction.
    objectId: int, default=-1
        Id assigned to hits produced by this camera.

    Stage Parameters
    ----------------
    mediumIdx: int
        Index of the medium the camera is submerged in
    radius: float, default=1.0
        Radius of the camera disk
    position: (float, float, float), default=(0.0,0.0,0.0)
        Position of the camera in world space
    direction: (float, float, float), default=(0.0,0.0,1.0)
        Direction the camera faces. Corresponds in local space positive z
        direction.
    up: (float, float, float), default=(0.0,1.0,0.0)
        Direction identifying where 'up' is for the camera. Corresponds in local
        space to the positive y direction.
    objectId: int, default=-1
        Id assigned to hits produced by this camera.

    Note
    ----
    `direction` and `up` may not be parallel.
    """

    name = "Disk Camera"

    class CameraParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("radius", c_float),
            ("objectId", c_int32),
            ("mediumIdx", c_uint32),
            ("_view", mat3),
        ]

    def __init__(
        self,
        *,
        mediumIdx: int,
        radius: float = 1.0 * u.m,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        up: tuple[float, float, float] = (0.0, 1.0, 0.0),
        objectId: int = -1,
    ) -> None:
        super().__init__(
            nRNGSamples=4,
            nRNGDirect=2,
            supportDirect=True,
            params={"CameraParams": self.CameraParams},
            extra={"direction", "up"},
        )
        # save params
        self.direction = direction
        self.up = up
        self.setParams(
            radius=radius,
            position=position,
            mediumIdx=mediumIdx,
            objectId=objectId,
        )

    @property
    def direction(self) -> tuple[float, float, float]:
        """Direction the camera faces"""
        return self._direction

    @direction.setter
    def direction(self, value: tuple[float, float, float]) -> None:
        self._direction = value

    @property
    def up(self) -> tuple[float, float, float]:
        """Direction of the local y-Axis identifying where 'up' us for the camera"""
        return self._up

    @up.setter
    def up(self, value: tuple[float, float, float]) -> None:
        self._up = value

    @property
    def sourceCode(self) -> str:
        return loadShader("camera/disk.glsl")

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)

        # create view matrix
        t = Transform.View(direction=self.getParam("direction"), up=self.getParam("up"))
        self.setParam("_view", t.innerMatrix)  # column major


class ConeCamera(Camera):
    """
    Sampling camera rays from a cone positioned at a single point.

    Parameters
    ----------
    mediumIdx: int
        Index of the medium the camera is submerged in.
    position: (float, float, float), default=(0.0,0.0,0.0)
        Cone position
    direction: (float, float, float), default=(0.0,0.0,1.0)
        Direction of the cone
    cosOpeningAngle: float, default=1.0
        Cosine of the cones opening angle
    objectId: int, default=-1
        Id assigned to hits produced by this camera.

    Stage Parameters
    ----------------
    mediumIdx: int
        Index of the medium the camera is submerged in.
    position: (float, float, float), default=(0.0,0.0,0.0)
        Cone position
    direction: (float, float, float), default=(0.0,0.0,1.0)
        Direction of the cone
    cosOpeningAngle: float, default=1.0
        Cosine of the cones opening angle
    objectId: int, default=-1
        Id assigned to hits produced by this camera.
    """

    name = "Cone Camera"

    class CameraParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("cosOpeningAngle", c_float),
            ("direction", vec3),
            ("mediumIdx", c_uint32),
            ("objectId", c_int32),
        ]

    def __init__(
        self,
        *,
        mediumIdx: int,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        cosOpeningAngle: float = 1.0,
        objectId: int = -1,
    ) -> None:
        super().__init__(
            nRNGSamples=2,
            nRNGDirect=0,
            supportDirect=True,
            params={"CameraParams": self.CameraParams},
        )
        self.setParams(
            position=position,
            direction=direction,
            cosOpeningAngle=cosOpeningAngle,
            mediumIdx=mediumIdx,
            objectId=objectId,
        )

    @property
    def sourceCode(self) -> str:
        return loadShader("camera/cone.glsl")


class SphereCamera(Camera):
    """
    Camera simulating an isotropic spherical detector of given radius accepting
    light from all visible directions at any position on it. Always uses a unit
    sphere in object space regardless of size.

    Parameters
    ----------
    mediumIdx: int
        Index of the medium the camera is submerged in.
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center position of the detector sphere
    radius: float, default=1.0
        Radius of the detector sphere
    timeDelta: float, default=0.0
        Time offset applied to camera rays and this light paths.
    objectId: int, default=-1
        Id assigned to hits produced by this camera.

    Stage Parameters
    ----------------
    mediumIdx: int
        Index of the medium the camera is submerged in.
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center position of the detector sphere
    radius: float, default=1.0
        Radius of the detector sphere
    timeDelta: float, default=0.0
        Time offset applied to camera rays and this light paths.
    objectId: int, default=-1
        Id assigned to hits produced by this camera.

    Note
    ----
    The sign of the radius determines wether the camera rays point outward
    (positive) or inward (negative).
    """

    name = "Spherical Camera"

    class CameraParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("radius", c_float),
            ("timeDelta", c_float),
            ("mediumIdx", c_uint32),
            ("objectId", c_int32),
            ("_contrib", c_float),
            ("_contribDirect", c_float),
        ]

    def __init__(
        self,
        *,
        mediumIdx: int,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 1.0,
        timeDelta: float = 0.0,
        objectId: int = -1,
    ) -> None:
        super().__init__(
            nRNGSamples=4,
            nRNGDirect=2,
            supportDirect=True,
            params={"CameraParams": self.CameraParams},
        )
        self.setParams(
            position=position,
            radius=radius,
            timeDelta=timeDelta,
            mediumIdx=mediumIdx,
            objectId=objectId,
        )

    @property
    def sourceCode(self) -> str:
        return loadShader("camera/sphere.glsl")

    def _finishParams(self, i: int) -> None:
        r = self.getParam("radius")
        contrib = 4 * np.pi * r**2 * 2 * np.pi
        contribDirect = 4 * np.pi * r**2
        self.setParam("_contrib", contrib)
        self.setParam("_contribDirect", contribDirect)


class PointCamera(Camera):
    """
    Camera producing rays isotropic in all directions from a single point.

    Parameters
    ----------
    mediumIdx: int
        Index of the medium the camera is submerged in.
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Origin of camera rays.
    timeDelta: float, default=0.0
        Time offset applied to camera rays.
    objectId: int, default=-1
        Id assigned to hits produced by this camera.

    Stage Parameters
    ----------------
    mediumIdx: int
        Index of the medium the camera is submerged in.
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Origin of camera rays.
    timeDelta: float, default=0.0
        Time offset applied to camera rays.
    objectId: int, default=-1
        Id assigned to hits produced by this camera.
    """

    name = "Point Camera"

    class CameraParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("timeDelta", c_float),
            ("mediumIdx", c_uint32),
            ("objectId", c_int32),
        ]

    def __init__(
        self,
        *,
        mediumIdx: int,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        timeDelta: float = 0.0,
        objectId: int = -1,
    ) -> None:
        super().__init__(
            nRNGSamples=2,
            supportDirect=False,
            params={"CameraParams": self.CameraParams},
        )
        self.setParams(
            position=position,
            timeDelta=timeDelta,
            mediumIdx=mediumIdx,
            objectId=objectId,
        )

    @property
    def sourceCode(self) -> str:
        return loadShader("camera/point.glsl")


class MeshCamera(Camera):
    """
    Camera producing rays at the surface of the given mesh.

    Parameters
    ----------
    mesh: MeshInstance
        Mesh to produce rays from
    mediumIdx: int
        Index of the medium the camera is submerged in.
    objectId: int, default=-1
        Id assigned to hits produced by this camera.
    timeDelta: float, default=0.0
        Time offset applied to camera rays.
    inward: bool, default=False
        If True, creates rays pointing inwards, i.e. opposite to the surface
        normal.

    Stage Parameters
    ----------------
    mesh: MeshInstance
        Mesh to produce rays from
    mediumIdx: int
        Index of the medium the camera is submerged in.
    objectId: int, default=-1
        Id assigned to hits produced by this camera.
    timeDelta: float, default=0.0
        Time offset applied to camera rays.
    inward: bool, default=False
        If True, creates rays pointing inwards, i.e. opposite to the surface
        normal.

    Note
    ----
    For the sampled rays the Mesh's normal will be flipped according to the
    inward flag, i.e. the normal of valid hits will always oppose the ray
    direction regardless of the outwards direction of the Mesh.
    """

    name = "Mesh Camera"

    class CameraParams(Structure):
        _fields_ = [
            ("_verticesAddress", buffer_reference),
            ("_indicesAddress", buffer_reference),
            ("_triangleCount", c_uint32),
            ("_outward", c_float),
            ("timeDelta", c_float),
            ("mediumIdx", c_uint32),
            ("objectId", c_int32),
            ("_objToWorld", mat4x3),
            ("_worldToObj", mat4x3),
        ]

    def __init__(
        self,
        mesh: MeshInstance,
        *,
        mediumIdx: int,
        objectId: int = -1,
        timeDelta: float = 0.0,
        inward: bool = False,
    ) -> None:
        super().__init__(
            nRNGSamples=5,
            nRNGDirect=3,
            supportDirect=True,
            params={"CameraParams": self.CameraParams},
            extra={"inward", "mesh"},
        )
        self.setParams(
            inward=inward,
            timeDelta=timeDelta,
            mesh=mesh,
            mediumIdx=mediumIdx,
            objectId=objectId,
        )

    @property
    def inward(self) -> bool:
        """Whether rays are created inward, i.e. opposite to surface normal"""
        return self._inward

    @inward.setter
    def inward(self, value: bool) -> None:
        self._inward = value
        self.setParam("_outward", -1.0 if value else 1.0)

    @property
    def mesh(self) -> MeshInstance:
        """Mesh from which rays are produced"""
        return self._mesh

    @mesh.setter
    def mesh(self, value: MeshInstance) -> None:
        self._mesh = value
        self.setParams(
            _verticesAddress=value.vertices,
            _indicesAddress=value.indices,
            _triangleCount=value.triangleCount,
            # GLSL expects column major -> transpose matrices
            _objToWorld=value.transform.numpy().T,
            _worldToObj=value.transform.inverse().numpy().T,
        )

    @property
    def sourceCode(self) -> str:
        return loadShader("camera/mesh.glsl")
