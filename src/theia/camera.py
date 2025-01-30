from __future__ import annotations

import numpy as np
import hephaistos as hp
from hephaistos.glsl import buffer_reference, mat3, mat4x3, vec3
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import QueueBuffer, QueueTensor, QueueView

from ctypes import Structure, c_float, c_uint32

import theia.units as u
from theia.light import WavelengthSampleItem, WavelengthSource
from theia.random import RNG
from theia.scene import MeshInstance, Transform
from theia.util import ShaderLoader, compileShader, createPreamble

from collections.abc import Callable


__all__ = [
    "Camera",
    "CameraRayItem",
    "CameraRaySampler",
    "ConeCamera",
    "FlatCamera",
    "HostCamera",
    "MeshCamera",
    "PencilCamera",
    "PointCamera",
    "PolarizedCameraRayItem",
    "SphereCamera",
]


def __dir__():
    return __all__


class Camera(SourceCodeMixin):
    """
    Base class for camera producing camera rays and samples used in backward,
    bidirectional and direct tracing.
    Running this stage inside a pipeline updates its usage in following stages.
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


class CameraRayItem(Structure):
    _fields_ = [
        ("position", c_float * 3),
        ("direction", c_float * 3),
        ("contrib", c_float),
        ("timeDelta", c_float),
        ("hitPosition", c_float * 3),
        ("hitDirection", c_float * 3),
        ("hitNormal", c_float * 3),
    ]


class PolarizedCameraRayItem(Structure):
    _fields_ = [
        ("position", c_float * 3),
        ("direction", c_float * 3),
        ("contrib", c_float),
        ("timeDelta", c_float),
        ("polarizationRef", c_float * 3),
        ("mueller", c_float * 16),
        ("hitPolRef", c_float * 3),
        ("hitPosition", c_float * 3),
        ("hitDirection", c_float * 3),
        ("hitNormal", c_float * 3),
    ]


class CameraRaySampler(PipelineStage):
    """
    Utility class for sampling a camera rays from the given `Camera` and
    storing the result in a queue.

    Parameters
    ----------
    camera: Camera
        Camera to sample
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from
    capacity: int
        Maximum number of samples that can be drawn per run
    rng: RNG | None, default=None
        The random number generator used for sampling. May be `None` if `camera`
        does not require random numbers.
    polarized: bool, default=True
        Whether to save polarization information.
    retrieve: bool, default=True
        Wether the queue gets retrieved from the device after sampling
    batchSize: int
        Number of samples to draw per run
    code: bytes | None, default=None
        Compiled source code. If `None`, the byte code get's compiled from
        source. Note, that the compiled code is not checked. You must ensure
        it matches the configuration.

    Stage Parameters
    ----------------
    count: int, default=capacity
        Number of samples to draw per run. Must be at most `capacity`.
    baseCount: int, default=0
        Offset into the sampling stream of the light source.
    """

    name = "Camera Ray Sampler"

    class SampleParams(Structure):
        _fields_ = [("count", c_uint32), ("baseCount", c_uint32)]

    def __init__(
        self,
        camera: Camera,
        wavelengthSource: WavelengthSource,
        capacity: int,
        *,
        rng: RNG | None = None,
        polarized: bool = True,
        retrieve: bool = True,
        batchSize: int = 128,
        code: bytes | None = None,
    ) -> None:
        # check if we have a rng if needed
        needRNG = camera.nRNGSamples > 0 or wavelengthSource.nRNGSamples > 0
        if needRNG and rng is None:
            raise ValueError("camera requires a rng but none was given!")
        # init stage
        super().__init__({"SampleParams": self.SampleParams})

        # save params
        self._batchSize = batchSize
        self._camera = camera
        self._capacity = capacity
        self._polarized = polarized
        self._retrieve = retrieve
        self._rng = rng
        self._wavelengthSource = wavelengthSource
        self.setParams(count=capacity, baseCount=0)

        # create code if needed
        if code is None:
            preamble = createPreamble(
                BATCH_SIZE=batchSize,
                CAMERA_QUEUE_SIZE=capacity,
                CAMERA_QUEUE_POLARIZED=polarized,
                PHOTON_QUEUE_SIZE=capacity,
                POLARIZATION=polarized,
            )
            headers = {
                "camera.glsl": camera.sourceCode,
                "rng.glsl": rng.sourceCode if rng is not None else "",
                "photon.glsl": wavelengthSource.sourceCode,
            }
            code = compileShader("camera.sample.glsl", preamble, headers)
        self._code = code
        self._program = hp.Program(self._code)
        # calculate group size
        self._groups = -(capacity // -batchSize)

        # create queue holding samples
        item = PolarizedCameraRayItem if polarized else CameraRayItem
        self._camTensor = QueueTensor(item, capacity)
        self._camBuffer = [
            QueueBuffer(item, capacity) if retrieve else None for _ in range(2)
        ]
        item = WavelengthSampleItem
        self._photonTensor = QueueTensor(item, capacity)
        self._photonBuffer = [
            QueueBuffer(item, capacity) if retrieve else None for _ in range(2)
        ]
        # bind memory
        self._program.bindParams(
            CameraQueueOut=self._camTensor,
            PhotonQueueOut=self._photonTensor,
        )

    @property
    def batchSize(self) -> int:
        """Number of samples drawn per work group"""
        return self._batchSize

    @property
    def camera(self) -> Camera:
        """Camera that is sampled"""
        return self._camera

    @property
    def capacity(self) -> int:
        """Maximum number of samples that can be drawn per run"""
        return self._capacity

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching"""
        return self._code

    @property
    def polarized(self) -> bool:
        """Whether to save polarization information"""
        return self._polarized

    @property
    def retrieve(self) -> bool:
        """Wether the queue gets retrieved from the device after sampling"""
        return self._retrieve

    @property
    def rng(self) -> RNG | None:
        """The random number generator used for sampling. None, if none used."""
        return self._rng

    @property
    def cameraTensor(self) -> QueueTensor:
        """Tensor holding the queue storing the samples"""
        return self._camTensor

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source used to sample wavelengths"""
        return self._wavelengthSource

    @property
    def wavelengthTensor(self) -> QueueTensor:
        """Tensor holding the sampled wavelengths"""
        return self._photonTensor

    def cameraBuffer(self, i: int) -> QueueBuffer | None:
        """
        Buffer holding the i-th queue containing camera samples.
        `None` if retrieve was set to False.
        """
        return self._camBuffer[i]

    def cameraView(self, i: int) -> QueueView | None:
        """
        View into the i-th queue containing camera samples.
        `None` if retrieve was set to `False`.
        """
        return self.cameraBuffer(i).view if self.retrieve else None

    def wavelengthBuffer(self, i: int) -> QueueBuffer | None:
        """
        Buffer holding the i-th queue containing wavelength samples.
        `None` if retrieve was set to False.
        """
        return self._photonBuffer[i]

    def wavelengthView(self, i: int) -> QueueView | None:
        """
        View into the i-th queue containing wavelength samples.
        `None` if retrieve was set to False.
        """
        return self.wavelengthBuffer(i).view if self.retrieve else None
    
    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all stages involved with this sampler in the correct
        order suitable for creating a pipeline.
        """
        stages = [] if self.rng is None else [self.rng]
        stages.extend([self.wavelengthSource, self.camera, self])
        return stages

    def run(self, i: int) -> list[hp.Command]:
        self._bindParams(self._program, i)
        self.camera.bindParams(self._program, i)
        self.wavelengthSource.bindParams(self._program, i)
        if self.rng is not None:
            self.rng.bindParams(self._program, i)
        cmds: list[hp.Command] = [self._program.dispatch(self._groups)]
        if self.retrieve:
            cmds.extend(
                [
                    hp.retrieveTensor(self.cameraTensor, self.cameraBuffer(i)),
                    hp.retrieveTensor(self.wavelengthTensor, self.wavelengthBuffer(i)),
                ]
            )
        return cmds


class HostCamera(Camera):
    """
    Camera passing camera ray samples from the CPU to the GPU.

    Parameters
    ----------
    capacity: int
        Maximum number of samples that can be drawn per run
    polarized: bool, default=False
        Whether the host also provides polarization information
    updateFn: (HostCamera, int) -> None | None, default=None
        Optional update function called before the pipeline processes a task.
        `i` is the i-th configuration the update should affect.
        Can be used to stream in new samples on demand.

    Note
    ----
    To comply with API requirements the shader code expects a wavelength,
    which gets ignored.
    """

    name = "Host Camera"

    _sourceCode = ShaderLoader("camera.host.glsl")

    def __init__(
        self,
        capacity: int,
        *,
        polarized: bool = False,
        updateFn: Callable[[HostCamera, int], None] | None = None,
    ) -> None:
        super().__init__(nRNGSamples=0, supportDirect=False)
        # save params
        self._capacity = capacity
        self._polarized = polarized
        self._updateFn = updateFn

        # allocate memory
        item = PolarizedCameraRayItem if polarized else CameraRayItem
        self._buffers = [
            QueueBuffer(item, capacity, skipCounter=True) for _ in range(2)
        ]
        self._tensor = QueueTensor(item, capacity, skipCounter=True)

    @property
    def capacity(self) -> int:
        """Maximum number of samples that can be drawn per run"""
        return self._capacity

    @property
    def polarized(self) -> bool:
        """Whether the host also provides polarization information"""
        return self._polarized

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            CAMERA_QUEUE_SIZE=self.capacity,
            CAMERA_QUEUE_POLARIZED=self.polarized,
        )
        return preamble + self._sourceCode

    def buffer(self, i: int) -> int:
        """Returns the i-th buffer containing the data for the next batch"""
        return self._buffers[i]

    def view(self, i: int) -> QueueView:
        """
        Returns a view of the data inside the i-th buffer
        """
        return self.buffer(i).view

    def bindParams(self, program: hp.Program, i: int) -> None:
        super().bindParams(program, i)
        program.bindParams(CameraQueueIn=self._tensor)

    # PipelineStage API

    def update(self, i: int) -> None:
        if self._updateFn is not None:
            self._updateFn(self, i)
        super().update(i)

    def run(self, i: int) -> list[hp.Command]:
        return [hp.updateTensor(self.buffer(i), self._tensor), *super().run(i)]


class PencilCamera(Camera):
    """
    Sampler outputting a constant camera ray with independent hit
    parameterization.

    Parameters
    ----------
    rayPosition: (float, float, float), default=(0.0, 0.0, 0.0)
        Start point of the camera ray
    rayDirection: (float, float, float), default=(0.0, 0.0, 1.0)
        Direction of the camera ray
    polarizationRef: (float, float, float)|None, default=None
        Reference frame of polarization indicating vertical polarized light.
        Must be unit and orthogonal to the ray direction.
        If None, creates an unspecified orthogonal one.
    timeDelta: float, default=0.0
        Extra time delay added on the ray
    hitPosition: (float, float, float), default=(0.0,0.0,0.0)
        Hit position in the local frame of the detector
    hitDirection: (float, float, float), default=(0.0,0.0,-1.0)
        Ray direction at the hit position in the local frame of the detector
    hitNormal: (float, float, float), default=(0.0,0.0,1.0)
        Surface normal at the hit position in the local frame of the detector
    hitPolarizationRef: (float, float, float)|None, default=None
        Reference frame of polarization indicating vertical polarized light
        defined in the object space of the hit/camera. Must be unit and
        orthogonal to the ray direction. If None, creates an unspecified
        orthogonal one.

    Stage Parameters
    ----------------
    rayPosition: (float, float, float), default=(0.0, 0.0, 0.0)
        Start point of the camera ray
    rayDirection: (float, float, float), default=(0.0, 0.0, 1.0)
        Direction of the camera ray
    polarizationRef: (float, float, float)
        Reference frame of polarization indicating vertical polarized light.
        Must be unit and orthogonal to the ray direction.
    timeDelta: float, default=0.0
        Extra time delay added on the ray
    hitPosition: (float, float, float), default=(0.0,0.0,0.0)
        Hit position in the local frame of the detector
    hitDirection: (float, float, float), default=(0.0,0.0,-1.0)
        Ray direction at the hit position in the local frame of the detector
    hitNormal: (float, float, float), default=(0.0,0.0,1.0)
        Surface normal at the hit position in the local frame of the detector
    hitPolarizationRef: (float, float, float)|None, default=None
        Reference frame of polarization indicating vertical polarized light
        defined in the object space of the hit/camera. Must be unit and
        orthogonal to the hit direction. If None, creates an unspecified
        orthogonal one.
    """

    name = "Pencil Camera Beam"

    class CameraParams(Structure):
        _fields_ = [
            ("rayPosition", vec3),
            ("rayDirection", vec3),
            ("polarizationRef", vec3),
            ("timeDelta", c_float),
            ("hitPosition", vec3),
            ("hitDirection", vec3),
            ("hitNormal", vec3),
            ("hitPolarizationRef", vec3),
        ]

    def __init__(
        self,
        *,
        rayPosition: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rayDirection: tuple[float, float, float] = (0.0, 0.0, 1.0),
        polarizationRef: tuple[float, float, float] | None = None,
        timeDelta: float = 0.0,
        hitPosition: tuple[float, float, float] = (0.0, 0.0, 0.0),
        hitDirection: tuple[float, float, float] = (0.0, 0.0, -1.0),
        hitNormal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        hitPolarizationRef: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__(
            nRNGSamples=0,
            supportDirect=False,
            params={"CameraParams": self.CameraParams},
        )
        # create polarization reference frame if not specified
        if polarizationRef is None:
            # take two cross products: with e_x and e_y
            # both cant be parallel to dir at the same time
            x, y, z = rayDirection
            ref1 = np.array([0.0, -z, y])
            ref2 = np.array([-z, 0.0, x])
            # take the longe one and normalize
            l1 = np.sqrt(np.square(ref1).sum())
            l2 = np.sqrt(np.square(ref2).sum())
            polarizationRef = tuple(ref1 / l1 if l1 > l2 else ref2 / l2)
        if hitPolarizationRef is None:
            x, y, z = hitDirection
            ref1 = np.array([0.0, -z, y])
            ref2 = np.array([-z, 0.0, x])
            # take the longe one and normalize
            l1 = np.sqrt(np.square(ref1).sum())
            l2 = np.sqrt(np.square(ref2).sum())
            hitPolarizationRef = tuple(ref1 / l1 if l1 > l2 else ref2 / l2)
        self.setParams(
            rayPosition=rayPosition,
            rayDirection=rayDirection,
            polarizationRef=polarizationRef,
            timeDelta=timeDelta,
            hitPosition=hitPosition,
            hitDirection=hitDirection,
            hitNormal=hitNormal,
            hitPolarizationRef=hitPolarizationRef,
        )

    # source code via descriptor
    sourceCode = ShaderLoader("camera.pencil.glsl")


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

    Stage Parameters
    ----------------
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

    Note
    ----
    `direction` and `up` may not be parallel.
    """

    name = "Flat Camera"

    class CameraParams(Structure):
        _fields_ = [
            ("width", c_float),
            ("length", c_float),
            ("position", vec3),
            ("_view", mat3),
        ]

    def __init__(
        self,
        *,
        width: float = 1.0 * u.cm,
        length: float = 1.0 * u.cm,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        up: tuple[float, float, float] = (0.0, 1.0, 0.0),
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
        self.setParams(width=width, length=length, position=position)

    # source code via descriptor
    sourceCode = ShaderLoader("camera.flat.glsl")

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
    position: (float, float, float), default=(0.0,0.0,0.0)
        Cone position
    direction: (float, float, float), default=(0.0,0.0,1.0)
        Direction of the cone
    cosOpeningAngle: float, default=1.0
        Cosine of the cones opening angle

    Stage Parameters
    ----------------
    position: (float, float, float), default=(0.0,0.0,0.0)
        Cone position
    direction: (float, float, float), default=(0.0,0.0,1.0)
        Direction of the cone
    cosOpeningAngle: float, default=1.0
        Cosine of the cones opening angle
    """

    name = "Cone Camera"

    class CameraParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("direction", vec3),
            ("cosOpeningAngle", c_float),
        ]

    def __init__(
        self,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        cosOpeningAngle: float = 1.0,
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
        )

    # source code via descriptor
    sourceCode = ShaderLoader("camera.cone.glsl")


class SphereCamera(Camera):
    """
    Camera simulating an isotropic spherical detector of given radius accepting
    light from all visible directions at any position on it. Always uses a unit
    sphere in object space regardless of size.

    Parameters
    ----------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center position of the detector sphere
    radius: float, default=1.0
        Radius of the detector sphere
    timeDelta: float, default=0.0
        Time offset applied to camera rays and this light paths.

    Stage Parameters
    ----------------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center position of the detector sphere
    radius: float, default=1.0
        Radius of the detector sphere
    timeDelta: float, default=0.0
        Time offset applied to camera rays and this light paths.

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
            ("_contrib", c_float),
            ("_contribDirect", c_float),
        ]

    def __init__(
        self,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 1.0,
        timeDelta: float = 0.0,
    ) -> None:
        super().__init__(
            nRNGSamples=4,
            nRNGDirect=2,
            supportDirect=True,
            params={"CameraParams": self.CameraParams},
        )
        self.setParams(position=position, radius=radius, timeDelta=timeDelta)

    # source code via descriptor
    sourceCode = ShaderLoader("camera.sphere.glsl")

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
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Origin of camera rays.
    timeDelta: float, default=0.0
        Time offset applied to camera rays.

    Stage Parameters
    ----------------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Origin of camera rays.
    timeDelta: float, default=0.0
        Time offset applied to camera rays.
    """

    name = "Point Camera"

    class CameraParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("timeDelta", c_float),
        ]

    def __init__(
        self,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        timeDelta: float = 0.0,
    ) -> None:
        super().__init__(
            nRNGSamples=2,
            supportDirect=False,
            params={"CameraParams": self.CameraParams},
        )
        self.setParams(position=position, timeDelta=timeDelta)

    # source code via descriptor
    sourceCode = ShaderLoader("camera.point.glsl")


class MeshCamera(Camera):
    """
    Camera producing rays at the surface of the given mesh.

    Parameters
    ----------
    mesh: MeshInstance
        Mesh to produce rays from
    timeDelta: float, default=0.0
        Time offset applied to camera rays.
    inward: bool, default=False
        If True, creates rays pointing inwards, i.e. opposite to the surface
        normal.

    Stage Parameters
    ----------------
    mesh: MeshInstance
        Mesh to produce rays from
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
            ("_objToWorld", mat4x3),
            ("_worldToObj", mat4x3),
        ]

    def __init__(
        self,
        mesh: MeshInstance,
        *,
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

    # source code via descriptor
    sourceCode = ShaderLoader("camera.mesh.glsl")
