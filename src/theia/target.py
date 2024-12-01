from __future__ import annotations

import numpy as np
from hephaistos.pipeline import SourceCodeMixin

from theia.scene import Transform
from theia.util import ShaderLoader
import theia.units as u

from ctypes import Structure, c_float
from hephaistos.glsl import mat3, vec3

__all__ = [
    "DiskTarget",
    "DiskTargetGuide",
    "FlatTarget",
    "FlatTargetGuide",
    "InnerSphereTarget",
    "SphereTarget",
    "SphereTargetGuide",
    "Target",
    "TargetGuide",
]


def __dir__():
    return __all__


class Target(SourceCodeMixin):
    """
    Base class for all target implementation. Targets are used in volume tracing
    as a substitute for an actual scene when estimating hits and shadowing. They
    provide three GPU functions:

    ```
    TargetSample sampleTarget(vec3 observer, [rng state])
    TargetSample intersectTarget(vec3 observer, vec3 direction)
    bool isOccludedByTarget(vec3 position)
    ```

    The first function samples the target given the position of an observer,
    whereas the second determines whether a ray shooting from a given position
    in a certain direction intersects the target.

    Targets are the equivalent to cameras in backward tracing.

    See Also
    --------
    theia.camera.Camera : Camera for backward tracing
    """

    name = "Target"

    def __init__(
        self,
        *,
        nRNGSamples: int,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self._nRNGSamples = nRNGSamples

    @property
    def nRNGSamples(self) -> int:
        """Amount of random numbers drawn per sample"""
        return self._nRNGSamples


class SphereTarget(Target):
    """
    Spherical target sampling the visible hemisphere. Hits in the local
    coordinate system are placed on the equivalent unit sphere at the origin.

    Parameters
    ----------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center of the sphere
    radius: float, default=1.0m
        Radius of the sphere

    Stage Parameters
    ----------------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center of the sphere
    radius: float, default=1.0m
        Radius of the sphere

    See Also
    --------
    theia.camera.SphereCamera : Camera equivalent for backward tracing
    theia.target.InnerSphereTarget : Sphere target sampled from inside
    """

    name = "Sphere Target"

    class TargetParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("radius", c_float),
            ("_invPos", vec3),
            ("_invRad", c_float),
            ("_hemisphereProb", c_float),
        ]

    def __init__(
        self,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 1.0 * u.m,
    ) -> None:
        super().__init__(nRNGSamples=2, params={"TargetParams": self.TargetParams})
        # save params
        self.setParams(
            position=position,
            radius=radius,
        )

    # source code via descriptor
    sourceCode = ShaderLoader("target.sphere.glsl")

    def _finishParams(self, i):
        r = self.getParam("radius")
        area = 2.0 * np.pi * r**2
        self.setParam("_hemisphereProb", 1.0 / area)
        # precalc stuff for worldToObj trafo
        x, y, z = self.position
        self.setParams(
            _invPos=(-x / r, -y / r, -z / r),
            _invRad=1.0 / r,
        )


class InnerSphereTarget(Target):
    """
    Spherical target sampled from the inside. Hits in the local coordinate
    system are placed on the equivalent unit sphere at the origin.

    Parameters
    ----------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center of the sphere
    radius: float, default=1.0m
        Radius of the sphere

    Stage Parameters
    ----------------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center of the sphere
    radius: float, default=1.0m
        Radius of the sphere

    See Also
    --------
    theia.target.SphereTarget : Sphere target sampled from outside
    """

    name = "Inner Sphere Target"

    class TargetParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("radius", c_float),
            ("_invPos", vec3),
            ("_invRad", c_float),
            ("_prob", c_float),
        ]

    def __init__(
        self,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 1.0 * u.m,
    ) -> None:
        super().__init__(nRNGSamples=2, params={"TargetParams": self.TargetParams})
        # save params
        self.setParams(position=position, radius=radius)

    # source code via descriptor
    sourceCode = ShaderLoader("target.sphere.inner.glsl")

    def _finishParams(self, i):
        r = self.getParam("radius")
        area = 4.0 * np.pi * r**2
        self.setParam("_prob", 1.0 / area)
        # precalc stuff for worldToObj trafo
        x, y, z = self.position
        self.setParams(
            _invPos=(-x / r, -y / r, -z / r),
            _invRad=1.0 / r,
        )


class FlatTarget(Target):
    """
    Rectangular target orientated freely in space denoted by a normal and up
    vector. Its sides are distinguished by the surface normal.

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

    See Also
    --------
    theia.camera.FlatCamera : Camera equivalent for backward tracing
    theia.target.DiskTarget : Similar but with spherical surface
    """

    name = "Flat Target"

    class TargetParams(Structure):
        _fields_ = [
            ("width", c_float),
            ("length", c_float),
            ("position", vec3),
            ("_normal", vec3),
            ("_prob", c_float),
            ("_objToWorld", mat3),
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
            nRNGSamples=2,
            params={"TargetParams": self.TargetParams},
            extra={"direction", "up"},
        )
        # save params
        self.direction = direction
        self.up = up
        self.setParams(width=width, length=length, position=position)

    # source code via descriptor
    sourceCode = ShaderLoader("target.flat.glsl")

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
        t = Transform.View(
            direction=self.getParam("direction"),
            up=self.getParam("up"),
        )
        m = t.innerMatrix
        self.setParam("_objToWorld", m.T)  # column major
        self.setParam("_normal", m[:, 2])

        area = self.width * self.length
        self.setParam("_prob", 1.0 / area)


class DiskTarget(Target):
    """
    Flat circular target orientated freely in space denoted by a normal vector.
    Its sides are distinguished by their surface normal.

    In the local coordinates of hits the circle lies in the xy plane centered in
    the origin with its normal facing in positive z direction.

    Parameters
    ----------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center of the circle
    radius: float, default=1.0m
        Radius of the circle
    normal: (float, float, float), default=(0.0, 0.0, 1.0)
        Normal vector of the disk corresponding to the z axis in local
        coordinates
    up: (float, float, float) | None, default=None
        Direction identifying where 'up' is for the target corresponding to the
        y axis in local coordinates.

    Stage Parameters
    ----------------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center of the circle
    radius: float, default=1.0m
        Radius of the circle
    normal: (float, float, float), default=(0.0, 0.0, 1.0)
        Normal vector of the disk corresponding to the z axis in local
        coordinates
    up: (float, float, float) | None, default=None
        Direction identifying where 'up' is for the target corresponding to the
        y axis in local coordinates.

    See Also
    --------
    theia.target.FlatCamera : Similar but with rectangular surface
    """

    name = "Disk Target"

    class TargetParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("radius", c_float),
            ("_normal", vec3),
            ("_prob", c_float),
            ("_objToWorld", mat3),
        ]

    def __init__(
        self,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 1.0 * u.m,
        normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        up: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__(
            nRNGSamples=2,
            params={"TargetParams": self.TargetParams},
            extra={"normal", "up"},
        )
        self.setParams(position=position, radius=radius, normal=normal, up=up)

    # source code via descriptor
    sourceCode = ShaderLoader("target.disk.glsl")

    @property
    def normal(self) -> tuple[float, float, float]:
        """Normal vector of the disk"""
        return self._normal

    @normal.setter
    def normal(self, value: tuple[float, float, float]) -> None:
        self._normal = value

    @property
    def up(self) -> tuple[float, float, float] | None:
        """Direction of the local y-Axis identifying where 'up' us for the target"""
        return self._up

    @up.setter
    def up(self, value: tuple[float, float, float] | None) -> None:
        self._up = value

    def _finishParams(self, i):
        super()._finishParams(i)

        t = Transform.View(
            direction=self.getParam("normal"),
            up=self.getParam("up"),
        )
        m = t.innerMatrix
        self.setParam("_objToWorld", m.T)  # column major
        self.setParam("_normal", m[:, 2])  # ensure normalized

        r = self.getParam("radius")
        area = np.pi * r**2
        self.setParam("_prob", 1.0 / area)


class TargetGuide(SourceCodeMixin):
    """
    Base class for all target guide implementation. Target guides are optionally
    used in forward scene tracer for multiple importance sampling the target by
    sampling directions we expect to reach the target. This is used to create
    alternative paths for each length by completing the path early in a process
    known as next event estimate.

    Unlike `Target`, the idea here is not to sample a point on the detector but
    a direction for a ray we expect to intersect the it. The corresponding
    intersection point is determined by the tracer. This allows target guides
    to be different from the detector shape.

    Target guides must provide two GPU functions:

    ```
    vec3 sampleTargetGuide(vec3 observer, [rng state], out float prob)
    float evalTargetGuide(vec3 observer, vec3 direction)
    ```

    The first function samples the target guide given the position of an
    observer returning the sampled direction, whereas the second determines
    the probability the given direction would have been sampled.
    """

    name = "Target Guide"

    def __init__(
        self,
        *,
        nRNGSamples: int,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self._nRNGSamples = nRNGSamples

    @property
    def nRNGSamples(self) -> int:
        """Amount of random numbers drawn per sample"""
        return self._nRNGSamples


class SphereTargetGuide(TargetGuide):
    """
    Spherical target guide sampling the hemisphere opposing the observer to
    ensure rays will travel completely through it.

    Parameters
    ----------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center of the sphere
    radius: float, default=1.0m
        Radius of the sphere

    Stage Parameters
    ----------------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center of the sphere
    radius: float, default=1.0m
        Radius of the sphere

    See Also
    --------
    theia.target.InnerSphereTargetGuide : Spherical guide sampled from within
    """

    name = "Sphere Target Guide"

    class TargetGuideParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("radius", c_float),
            ("_prob", c_float),
        ]

    def __init__(
        self,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 1.0 * u.m,
    ) -> None:
        super().__init__(
            nRNGSamples=2,
            params={"TargetGuideParams": self.TargetGuideParams},
        )
        # save params
        self.setParams(
            position=position,
            radius=radius,
        )

    # source code via descriptor
    sourceCode = ShaderLoader("target_guide.sphere.glsl")

    def _finishParams(self, i):
        r = self.getParam("radius")
        area = 2.0 * np.pi * r**2
        self.setParam("_prob", 1.0 / area)


class FlatTargetGuide(TargetGuide):
    """
    Rectangular target guide orientated freely in space denoted by a normal and
    up vector. Only samples the outside face, that is the side the normal vector
    points to. If the outside face is not visible at the path's current position
    it will indicate to the tracer to skip the next event estimate.

    Parameters
    ----------
    width: float, default=1.0m
        Width of the detector. Extent in the direction orthogonal to `normal`
        and `up`.
    height: float, default=1.0m
        Height of the detector. Extent in the `up` direction.
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Position of the rectangle's center.
    normal: (float, float, float), default=(0.0, 0.0, 1.0)
        Normal vector of the rectangle specifying the outside face.
    up: (float, float, float), default=(0.0, 1.0, 0.0)
        Direction specifying the height dimension.

    Stage Parameters
    ----------------
    width: float, default=1.0m
        Width of the detector. Extent in the direction orthogonal to `normal`
        and `up`.
    height: float, default=1.0m
        Height of the detector. Extent in the `up` direction.
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Position of the rectangle's center.
    normal: (float, float, float), default=(0.0, 0.0, 1.0)
        Normal vector of the rectangle specifying the outside face.
    up: (float, float, float), default=(0.0, 1.0, 0.0)
        Direction specifying the height dimension.

    Note
    ----
    `direction` and `up` may not be parallel.

    See Also
    --------
    theia.target.FlatTarget : Similar target for volume tracing
    """

    name = "Flat Target Guide"

    class TargetGuideParams(Structure):
        _fields_ = [
            ("width", c_float),
            ("height", c_float),
            ("position", vec3),
            ("_normal", vec3),
            ("_prob", c_float),
            ("_objToWorld", mat3),
        ]

    def __init__(
        self,
        *,
        width: float = 1.0 * u.m,
        height: float = 1.0 * u.m,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        up: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        super().__init__(
            nRNGSamples=2,
            params={"TargetGuideParams": self.TargetGuideParams},
            extra={"normal", "up"},
        )
        # save params
        self.normal = normal
        self.up = up
        self.setParams(width=width, height=height, position=position)

    # source code via descriptor
    sourceCode = ShaderLoader("target_guide.flat.glsl")

    @property
    def normal(self) -> tuple[float, float, float]:
        """Normal of the target guide"""
        return self._normal

    @normal.setter
    def normal(self, value: tuple[float, float, float]) -> None:
        self._normal = value

    @property
    def up(self) -> tuple[float, float, float]:
        """Up direction denoting the height dimension"""
        return self._up

    @up.setter
    def up(self, value: tuple[float, float, float]) -> None:
        self._up = value

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)

        t = Transform.View(
            direction=self.getParam("normal"),
            up=self.getParam("up"),
        )
        m = t.innerMatrix
        self.setParam("_objToWorld", m.T)  # column major
        self.setParam("_normal", m[:, 2])

        area = self.width * self.height
        self.setParam("_prob", 1.0 / area)


class DiskTargetGuide(TargetGuide):
    """
    Flat circular target guide orientated freely in space donated by a normal
    vector and a radius. Only samples the outside face, that is the side the
    normal vector points to. If the outside face is not visible at the path's
    current position it will indicate to the tracer to skip the next event
    estimate.

    Parameters
    ----------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center of the disk
    radius: float, default=1.0m
        Radius of the circle
    normal: (float, float, float), default(0.0, 0.0, 1.0)
        Normal vector of the disk specifying the outside face.

    Stage Parameters
    ----------------
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Center of the disk
    radius: float, default=1.0m
        Radius of the circle
    normal: (float, float, float), default(0.0, 0.0, 1.0)
        Normal vector of the disk specifying the outside face.

    See Also
    --------
    theia.target.DiskTarget : Similar target for volume tracing
    """

    name = "Disk Target Guide"

    class TargetGuideParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("radius", c_float),
            ("_normal", vec3),
            ("_prob", c_float),
            ("_objToWorld", mat3),
        ]

    def __init__(
        self,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 1.0 * u.m,
        normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        super().__init__(
            nRNGSamples=2,
            params={"TargetGuideParams": self.TargetGuideParams},
            extra={"normal"},
        )
        # save params
        self.setParams(
            position=position,
            radius=radius,
            normal=normal,
        )

    # source code via descriptor
    sourceCode = ShaderLoader("target_guide.disk.glsl")

    @property
    def normal(self) -> tuple[float, float, float]:
        """Normal of the target guide"""
        return self._normal

    @normal.setter
    def normal(self, value: tuple[float, float, float]) -> None:
        self._normal = value

    def _finishParams(self, i):
        r = self.getParam("radius")
        area = np.pi * r**2
        self.setParam("_prob", 1.0 / area)

        # assemble view matrix
        def normalize(name):
            v = np.array(self.getParam(name))
            return v / np.sqrt(np.square(v).sum(-1))

        z = normalize("normal")
        self.setParam("_normal", z)

        # create cosy from normal vector
        # see PBRT 4th edition by M. Pharr et al., ch. 3.3.3

        vx, vy, vz = z
        sign = np.copysign(1.0, vz)
        a = -1.0 / (sign + vz)
        b = vx * vy * a
        x = np.array([1.0 + sign * vx**2 * a, sign * b, -sign * vx])
        y = np.array([b, sign + vy**2 * a, -vy])
        mat = np.stack([x, y, z], -1)
        self.setParam("_objToWorld", mat)
