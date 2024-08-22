from __future__ import annotations

import numpy as np
import hephaistos as hp
from hephaistos.glsl import mat3, vec3
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import QueueBuffer, QueueTensor, QueueView

from ctypes import Structure, c_float, c_uint32

import theia.units as u
from theia.random import RNG
from theia.scene import Transform
from theia.util import ShaderLoader, compileShader, createPreamble

from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

__all__ = [
    "CameraRaySampler",
    "CameraRaySource",
    "ConeCameraRaySource",
    "FlatCameraRaySource",
    "HostCameraRaySource",
    "LenseCameraRaySource",
    "PencilCameraRaySource",
    "PointCameraRaySource",
    "PolarizedCameraRayItem",
]


def __dir__():
    return __all__


class CameraRaySource(SourceCodeMixin):
    """
    Base class for camera ray sources used in bidirectional path tracing.
    Running this stage inside a pipeline updates its usage in following stages.
    """

    name = "Camera Ray Source"

    def __init__(
        self,
        *,
        nRNGSamples: int,
        params: Dict[str, Type[Structure]] = {},
        extra: Set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self._nRNGSamples = nRNGSamples

    @property
    def nRNGSamples(self) -> int:
        """Amount of random numbers drawn per sample"""
        return self._nRNGSamples


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
        ("hitPosition", c_float * 3),
        ("hitDirection", c_float * 3),
        ("hitNormal", c_float * 3),
    ]


class CameraRaySampler(PipelineStage):
    """
    Utility class for sampling a `CameraRaySource` and storing the result in a
    queue.

    Parameters
    ----------
    camera: CameraRaySource
        Camera to sample
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
        camera: CameraRaySource,
        capacity: int,
        *,
        rng: Optional[RNG] = None,
        polarized: bool = True,
        retrieve: bool = True,
        batchSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        # check if we have a rng if needed
        if camera.nRNGSamples > 0 and rng is None:
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
        self.setParams(count=capacity, baseCount=0)

        # create code if needed
        if code is None:
            preamble = createPreamble(
                BATCH_SIZE=batchSize,
                CAMERA_QUEUE_SIZE=capacity,
                CAMERA_QUEUE_POLARIZED=polarized,
                POLARIZATION=polarized,
            )
            headers = {
                "camera.glsl": camera.sourceCode,
                "rng.glsl": rng.sourceCode if rng is not None else "",
            }
            code = compileShader("camera.sample.glsl", preamble, headers)
        self._code = code
        self._program = hp.Program(self._code)
        # calculate group size
        self._groups = -(capacity // -batchSize)

        # create queue holding samples
        item = PolarizedCameraRayItem if polarized else CameraRayItem
        self._tensor = QueueTensor(item, capacity)
        self._buffer = [
            QueueBuffer(item, capacity) if retrieve else None for _ in range(2)
        ]
        # bind memory
        self._program.bindParams(CameraQueueOut=self._tensor)

    @property
    def batchSize(self) -> int:
        """Number of samples drawn per work group"""
        return self._batchSize

    @property
    def camera(self) -> CameraRaySource:
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
    def rng(self) -> Optional[RNG]:
        """The random number generator used for sampling. None, if none used."""
        return self._rng

    @property
    def tensor(self) -> QueueTensor:
        """Tensor holding the queue storing the samples"""
        return self._tensor

    def buffer(self, i: int) -> Optional[QueueBuffer]:
        """
        Buffer holding the i-th queue. `None` if retrieve was set to False.
        """
        return self._buffer[i]

    def view(self, i: int) -> Optional[QueueView]:
        """View into the i-th queue. `None` if retrieve was set to `False`."""
        return self.buffer(i).view if self.retrieve else None

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        self.camera.bindParams(self._program, i)
        if self.rng is not None:
            self.rng.bindParams(self._program, i)
        cmds: List[hp.Command] = [self._program.dispatch(self._groups)]
        if self.retrieve:
            cmds.append(hp.retrieveTensor(self.tensor, self.buffer(i)))
        return cmds


class HostCameraRaySource(CameraRaySource):
    """
    Camera ray source passing samples from the CPU to the GPU.

    Parameters
    ----------
    capacity: int
        Maximum number of samples that can be drawn per run
    polarized: bool, default=False
        Whether the host also provides polarization information
    updateFn: (HostCameraRaySource, int) -> None | None, default=None
        Optional update function called before the pipeline processes a task.
        `i` is the i-th configuration the update should affect.
        Can be used to stream in new samples on demand.
    """

    name = "Host Camera Ray Source"

    _sourceCode = ShaderLoader("camera.host.glsl")

    def __init__(
        self,
        capacity: int,
        *,
        polarized: bool = False,
        updateFn: Optional[Callable[[HostCameraRaySource, int], None]] = None,
    ) -> None:
        super().__init__(nRNGSamples=0)
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

    def run(self, i: int) -> List[hp.Command]:
        return [hp.updateTensor(self.buffer(i), self._tensor), *super().run(i)]


class PencilCameraRaySource(CameraRaySource):
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
    """

    name = "Pencil Camera Beam"

    class CameraRayParams(Structure):
        _fields_ = [
            ("rayPosition", vec3),
            ("rayDirection", vec3),
            ("polarizationRef", vec3),
            ("timeDelta", c_float),
            ("hitPosition", vec3),
            ("hitDirection", vec3),
            ("hitNormal", vec3),
        ]

    def __init__(
        self,
        *,
        rayPosition: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rayDirection: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        polarizationRef: Union[Tuple[float, float, float], None] = None,
        timeDelta: float = 0.0,
        hitPosition: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        hitDirection: Tuple[float, float, float] = (0.0, 0.0, -1.0),
        hitNormal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        super().__init__(
            nRNGSamples=0, params={"CameraRayParams": self.CameraRayParams}
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
        self.setParams(
            rayPosition=rayPosition,
            rayDirection=rayDirection,
            polarizationRef=polarizationRef,
            timeDelta=timeDelta,
            hitPosition=hitPosition,
            hitDirection=hitDirection,
            hitNormal=hitNormal,
        )

    # source code via descriptor
    sourceCode = ShaderLoader("camera.pencil.glsl")


class FlatCameraRaySource(CameraRaySource):
    """
    Camera ray source simulating a rectangle as detector surface, i.e. samples
    first a point on the rectangle followed by a random direction in the upper
    hemisphere.

    In the local coordinates system of hits the rectangle lies in the xy plane
    centered in the origin with its normal facing in positive z direction. Width
    is its size in the x direction, length in the y direction. A `Transform` is
    applied to translate from local to global coordinates, thus allowing the
    rectangle to be positioned in the scene.

    Parameters
    ----------
    width: float
        Width of the detector
    length: float
        Length of the detector
    transform: Transform, default=identity
        Transformation applied to change from local to scene coordinates

    Stage Parameters
    ----------------
    width: float, default=1cm
        Width of the detector
    length: float, default=1cm
        Length of the detector
    transform: Transform, default=identity
        Transformation applied to change from local to scene coordinates

    Note
    ----
    Non orthogonal transformation are not handled correctly and will raise an
    exception.
    """

    name = "Flat Camera Ray Source"

    class CameraRayParams(Structure):
        _fields_ = [
            ("width", c_float),
            ("length", c_float),
            ("_contrib", c_float),
            ("_offset", vec3),
            ("_mat", mat3),
        ]

    def __init__(
        self,
        *,
        width: float = 1.0 * u.cm,
        length: float = 1.0 * u.cm,
        transform: Transform = Transform(),
    ) -> None:
        super().__init__(
            nRNGSamples=4,
            params={"CameraRayParams": self.CameraRayParams},
            extra={"transform"},
        )
        # save params
        self.transform = transform
        self.setParams(width=width, length=length)

    # source code via descriptor
    sourceCode = ShaderLoader("camera.flat.glsl")

    @property
    def transform(self) -> Transform:
        """
        Transformation applied on the camera to change from local to scene
        coordinates.
        """
        return self._transform

    @transform.setter
    def transform(self, value: Transform) -> None:
        self._transform = value
        mat = value.numpy()
        # we dont take scaling/shearing into account
        # generate warning if transform is not orthogonal
        det = np.linalg.det(mat[:, :3])
        if np.abs(np.abs(det) - 1.0) > 1e-5:
            raise ValueError("transform is not orthogonal!")

        self.setParam("_offset", mat[:, 3])
        self.setParam("_mat", mat[:, :3].T)  # row major -> column major

    def _finishParams(self, i: int) -> None:
        # p = 1 / (area * 2pi) -> contrib = area * 2pi
        # TODO: Not totally certain on wether we need to change p(point) from
        #       point space to directional space (i.e. a factor cos/r^2)
        area = self.getParam("width") * self.getParam("length")
        self.setParam("_contrib", 2.0 * np.pi * area)  # 1 / prob


class ConeCameraRaySource(CameraRaySource):
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

    name = "Cone Camera Ray Source"

    class CameraRayParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("direction", vec3),
            ("cosOpeningAngle", c_float),
        ]

    def __init__(
        self,
        *,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        cosOpeningAngle: float = 1.0,
    ) -> None:
        super().__init__(
            nRNGSamples=2,
            params={"CameraRayParams": self.CameraRayParams},
        )
        self.setParams(
            position=position,
            direction=direction,
            cosOpeningAngle=cosOpeningAngle,
        )

    # source code via descriptor
    sourceCode = ShaderLoader("camera.cone.glsl")


class LenseCameraRaySource(CameraRaySource):
    """
    Camera ray source similar to `FlatCameraRaySource` simulating a rectangular
    detector, but is illuminated exclusively through a lense in front of it.
    The lense itself is not simulated, but rays are restricted to hitting it.

    Source is defined in local detector coordinates. The detector rectangle lies
    on the xy plane centered in the origin with width being the dimension in x
    direction and length the dimension in y direction. The lense is parallel to
    the detector offset in z direction by a distance `focalLength` and modeled
    as circle of given radius.

    Parameters
    ----------
    width: float, default=1cm
        Width of the detector
    length: float, default=1cm
        Length of the detector
    focalLength: float, default=10cm
        Distance between detector and lense
    lenseRadius: float, default=5cm
        Radius of the lense
    transform: Transform, default=identity
        Transformation applied to change from local to scene coordinates

    Stage Parameters
    ----------------
    width: float, default=1cm
        Width of the detector
    length: float, default=1cm
        Length of the detector
    focalLength: float, default=10cm
        Distance between detector and lense
    lenseRadius: float, default=5cm
        Radius of the lense
    transform: Transform, default=identity
        Transformation applied to change from local to scene coordinates

    Note
    ----
    Non orthogonal transformation are not handled correctly and will generate
    a warning.
    """

    name = "Lense Camera Ray Source"

    class CameraRayParams(Structure):
        _fields_ = [
            ("width", c_float),
            ("length", c_float),
            ("focalLength", c_float),
            ("lenseRadius", c_float),
            ("_contrib", c_float),
            ("_offset", vec3),
            ("_mat", mat3),
        ]

    def __init__(
        self,
        *,
        width: float = 1.0 * u.cm,
        length: float = 1.0 * u.cm,
        focalLength: float = 10.0 * u.cm,
        lenseRadius: float = 5.0 * u.cm,
        transform: Transform = Transform(),
    ) -> None:
        super().__init__(
            nRNGSamples=4,
            params={"CameraRayParams": self.CameraRayParams},
            extra={"transform"},
        )
        # save params
        self.transform = transform
        self.setParams(
            width=width,
            length=length,
            focalLength=focalLength,
            lenseRadius=lenseRadius,
        )

    # source code via descriptor
    sourceCode = ShaderLoader("camera.lense.glsl")

    @property
    def transform(self) -> Transform:
        """
        Transformation applied on the camera to change from local to scene
        coordinates.
        """
        return self._transform

    @transform.setter
    def transform(self, value: Transform) -> None:
        self._transform = value
        mat = value.numpy()
        # we dont take scaling/shearing into account
        # generate warning if transform is not orthogonal
        det = np.linalg.det(mat[:, :3])
        if np.abs(np.abs(det) - 1.0) > 1e-5:
            raise ValueError("transform is not orthogonal!")

        self.setParam("_offset", mat[:, 3])
        self.setParam("_mat", mat[:, :3].T)  # row major -> column major

    def _finishParams(self, i: int) -> None:
        # p(sample) = 1/(A_det * A_lense)
        # p(dir) = p(sample) * cos(theta)/d^2 = p(sample) * focal/d^3
        # where theta = angle to lense normal; d distance point det <-> point lense
        # contrib = 1/p
        area_det = self.getParam("width") * self.getParam("length")
        area_lense = np.pi * self.getParam("lenseRadius") ** 2
        contrib = area_det * area_lense / self.getParam("focalLength")
        self.setParam("_contrib", contrib)


class SphereCameraRaySource(CameraRaySource):
    """
    Camera ray source simulating an isotropic spherical detector of given
    radius. Always uses a unit sphere in object space regardless of size.

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
    """

    name = "Spherical Camera Ray Source"

    class CameraRayParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("radius", c_float),
            ("timeDelta", c_float),
            ("_contrib", c_float),
        ]

    def __init__(
        self,
        *,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 1.0,
        timeDelta: float = 0.0,
    ) -> None:
        super().__init__(
            nRNGSamples=4,
            params={"CameraRayParams": self.CameraRayParams},
        )
        self.setParams(position=position, radius=radius, timeDelta=timeDelta)

    # source code via descriptor
    sourceCode = ShaderLoader("camera.sphere.glsl")

    def _finishParams(self, i: int) -> None:
        r = self.getParam("radius")
        contrib = 4 * np.pi * r**2 * 2 * np.pi
        self.setParam("_contrib", contrib)


class PointCameraRaySource(CameraRaySource):
    """
    Camera ray source producing rays isotropic in all directions from a single
    point.

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

    name = "Point Camera Ray Source"

    class CameraRayParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("timeDelta", c_float),
        ]

    def __init__(
        self,
        *,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        timeDelta: float = 0.0,
    ) -> None:
        super().__init__(
            nRNGSamples=2, params={"CameraRayParams": self.CameraRayParams}
        )
        self.setParams(position=position, timeDelta=timeDelta)

    # source code via descriptor
    sourceCode = ShaderLoader("camera.point.glsl")
