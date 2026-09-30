from __future__ import annotations

import numpy as np
import hephaistos as hp
from hephaistos.glsl import vec3
from ctypes import Structure, c_float

import itertools
import trimesh

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from theia.material import MaterialFlags, MaterialStore
import theia.units as u

from collections.abc import Iterable, Mapping, Sequence
from numpy.typing import NDArray, ArrayLike
from typing import Final


__all__ = [
    "linkPortals",
    "loadMesh",
    "MeshInstance",
    "MeshStore",
    "MultiScene",
    "RectBBox",
    "Scene",
    "SceneBase",
    "SceneTemplate",
    "SphereBBox",
    "Transform",
]


def __dir__():
    return __all__


class Transform:
    """Util class for creating transformation matrices"""

    def __init__(self, matrix: NDArray[np.float32] | None = None) -> None:
        self._arr = np.identity(4)
        if matrix is not None:
            # ensure it is an array
            matrix = np.asarray_chkfinite(matrix)
            # check shape
            if matrix.shape != (3, 4):
                raise ValueError("matrix must be of shape (3,4)!")
            self._arr[:3, :] = matrix

    def apply(self, points: ArrayLike) -> NDArray:
        """Applies the transformation to the given points of shape (N,3)"""
        return np.asarray(points) @ self._arr[:3, :3].T + self._arr[:3, 3]

    def applyVec(self, vector: ArrayLike) -> NDArray:
        """
        Applies the transformation to the given vectors of shape (N,3).
        Similar to `apply`, but translation are ignored.
        """
        return np.asarray(vector) @ self._arr[:3, :3].T

    def copy(self) -> Transform:
        """Creates a new independent copy of this transformation"""
        return Transform(self._arr[:3, :].copy())

    def freeze(self) -> Transform:
        """Prevents the transform from being changed"""
        self._arr.setflags(write=False)
        return self

    def inverse(self) -> Transform:
        """Returns the inverse transformation"""
        inv = Transform()
        inv._arr = np.linalg.inv(self._arr)
        return inv

    def numpy(self) -> NDArray:
        """
        Returns a numpy array in the correct format to be used with mesh
        instances
        """
        return np.ascontiguousarray(self._arr[:3, :], dtype=np.float32)

    @property
    def innerMatrix(self) -> NDArray:
        """Returns the inner 3x3 transformation matrix"""
        return self.numpy()[:3, :3]

    @property
    def offset(self) -> NDArray:
        """Returns the translation part of the transformation"""
        return self.numpy()[:3, 3]

    @staticmethod
    def Rotation(dx: float, dy: float, dz: float, angle: float) -> Transform:
        """
        Returns a rotation transformation around the axis (dx,dy,dz)
        counter-clockwise for the given angle.
        """
        # normalize unit direction
        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        dx /= length
        dy /= length
        dz /= length
        # create k matrix
        K = np.array([[0.0, -dz, dy], [dz, 0.0, -dx], [-dy, dx, 0.0]])
        # create rotation matrix
        res = Transform()
        angle = u.convert(angle, u.rad)
        res._arr[:3, :3] += np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        return res

    def rotate(self, dx: float, dy: float, dz: float, angle: float) -> Transform:
        """
        Returns a copy of the transformation rotated around (dx,dy,dz)
        counter-clockwise for the given angle.
        """
        return Transform.Rotation(dx, dy, dz, angle) @ self

    @staticmethod
    def Scale(x: float, y: float | None = None, z: float | None = None) -> Transform:
        """
        Returns a scale transformation using either a common factor for all axis
        or a distinct one for each individually.
        """
        res = Transform()
        res._arr[0, 0] = x
        res._arr[1, 1] = y if y is not None else x
        res._arr[2, 2] = z if z is not None else x
        return res

    def scale(
        self, x: float, y: float | None = None, z: float | None = None
    ) -> Transform:
        """
        Scales the transformation either by a common factor or in each dimension
        independently and returns the new transformation without altering the
        existing one.
        """
        return Transform.Scale(x, y, z) @ self

    @staticmethod
    def Translation(x: float, y: float, z: float) -> Transform:
        """Returns a translation transformation"""
        res = Transform()
        res._arr[:3, -1] = (x, y, z)
        return res

    def translate(self, x: float, y: float, z: float) -> Transform:
        """Returns a copy of the transform translated by the given amount"""
        return Transform.Translation(x, y, z) @ self

    @staticmethod
    def TRS(
        *,
        scale: tuple[float, float, float] | float = 1.0,
        rotate: tuple[float, float, float, float] | None = None,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Transform:
        """
        Shorthand function for creating a new transform using the given
        scale, rotation and translation applied in that order.

        Parameters
        ----------
        scale: (float, float, float) | float, default=1.0
            Either a common scaling factor for all axis or a distinct one for
            each.
        rotate: (float, float, float, float) | None, default=None
            Optional rotation. First three elements define the rotation axis,
            the last the rotation angle. Rotates counter-clockwise.
        translate: (float, float, float), default=(0.0, 0.0, 0.0)
            Translation given as an offset added.
        """
        result = Transform()
        if isinstance(scale, tuple):
            result = result.scale(*scale)
        else:
            result = result.scale(scale)
        if rotate is not None:
            result = result.rotate(*rotate)
        result = result.translate(*translate)
        return result

    @staticmethod
    def View(
        *,
        position: tuple[float, float, float] | NDArray = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] | NDArray = (0.0, 0.0, 1.0),
        up: tuple[float, float, float] | NDArray = (0.0, 1.0, 0.0),
    ) -> Transform:
        """
        Creates the view matrix mimicking a orthogonal camera at a specified
        position pointing in a given direction. Assumes in object space that the
        camera points in positive z direction and its up directions to align
        with the y axis.

        Parameters
        ----------
        position: (float, float, float) | NDArray, default=(0.0, 0.0, 0.0)
            Position of the camera
        direction: (float, float, float) | NDArray, default=(0.0, 0.0, 1.0)
            Direction the camera points
        up: (float, float, float) | NDArray, default=(0.0, 1.0, 0.0)
            Direction aligning with the camera's up direction

        Note
        ----
        `direction` and `up` may not be parallel

        See Also
        --------
        theia.scene.Transform.LookAt : View transform specified with a target
        """
        # ensure params are numpy arrays
        position = np.array(position)
        direction = np.array(direction)
        up = np.array(up)

        # calculate new coordinate system
        norm = lambda v: v / np.sqrt(np.square(v).sum(-1))
        z = norm(direction)
        x = norm(np.cross(up, direction))
        y = norm(np.cross(z, x))  # just to be safe
        # assemble matrix
        m = np.stack([x, y, z, position], -1)
        return Transform(m)

    @staticmethod
    def LookAt(
        *,
        position: tuple[float, float, float] | NDArray = (0.0, 0.0, 0.0),
        target: tuple[float, float, float] | NDArray = (0.0, 0.0, 1.0),
        up: tuple[float, float, float] | NDArray = (0.0, 1.0, 0.0),
    ) -> Transform:
        """
        Creates a transformation mimicking a orthogonal camera put at a
        specified position pointing at a target. Assumes in object space that
        the camera points in positive z direction and that its up direction
        aligns with the y axis.

        Parameters
        ----------
        position: (float, float, float), default=(0.0, 0.0, 0.0)
            Position of the camera
        target: (float, float, float), default=(0.0, 0.0, 1.0),
            Target the camera points at.
        up: (float, float, float), default=(0.0, 1.0, 0.0)
            Direction aligning with the cameras up direction.

        Note
        ----
        `up` may not point from `position` to `target`.

        See Also
        --------
        theia.scene.Transform.View : Creates a view transformation
        """
        # ensure params are numpy arrays
        position = np.array(position)
        target = np.array(target)
        up = np.array(up)

        # create transform
        return Transform.View(
            position=position,
            direction=(position - target),
            up=up,
        )

    def __matmul__(self, other: Transform) -> Transform:
        if type(other) != Transform:
            raise TypeError(other)
        result = Transform()
        result._arr = self._arr @ other._arr
        return result

    def __imatmul__(self, other: Transform) -> Transform:
        if type(other) != Transform:
            raise TypeError(other)
        if not self._arr.flags.writeable:
            raise RuntimeError("Tried to change frozen Transform!")
        self._arr = other._arr @ self._arr
        return self

    def __repr__(self):
        pre = "Transform("
        txt = np.array2string(self.numpy(), separator=",", prefix=pre, suffix=")")
        return pre + txt + ")"

    def __str__(self):
        return np.array_str(self.numpy(), precision=5, suppress_small=True)


class RectBBox:
    """
    Rectangular bounding box defined by two opposite corners

    Parameter
    ---------
    lowerCorner: tuple[float, float, float]
        The corner with minimal coordinate values

    upperCorner: tuple[float, float, float]
        The corner with maximal coordinate values
    """

    class GLSL(Structure):
        """GLSL struct equivalent"""

        _fields_ = [("lowerCorner", vec3), ("upperCorner", vec3)]

    def __init__(
        self,
        lowerCorner: tuple[float, float, float],
        upperCorner: tuple[float, float, float],
    ) -> None:
        self._glsl = self.GLSL()
        self.lowerCorner = lowerCorner
        self.upperCorner = upperCorner

    @property
    def glsl(self) -> RectBBox.GLSL:
        """The underlying GLSL structure to be consumed by shaders"""
        return self._glsl

    @property
    def diagonal(self) -> float:
        """Length of the box' diagonal, i.e. the distance between the two corners"""
        d = np.subtract(self.upperCorner, self.lowerCorner)
        return np.sqrt(np.square(d).sum())

    @property
    def lowerCorner(self) -> tuple[float, float, float]:
        """The corner with minimal coordinate values"""
        return (
            self._glsl.lowerCorner.x,
            self._glsl.lowerCorner.y,
            self._glsl.lowerCorner.z,
        )

    @lowerCorner.setter
    def lowerCorner(self, value: tuple[float, float, float]) -> None:
        self._glsl.lowerCorner.x = value[0]
        self._glsl.lowerCorner.y = value[1]
        self._glsl.lowerCorner.z = value[2]

    @property
    def upperCorner(self) -> tuple[float, float, float]:
        """The corner with maximal coordinate values"""
        return (
            self._glsl.upperCorner.x,
            self._glsl.upperCorner.y,
            self._glsl.upperCorner.z,
        )

    @upperCorner.setter
    def upperCorner(self, value: tuple[float, float, float]) -> None:
        self._glsl.upperCorner.x = value[0]
        self._glsl.upperCorner.y = value[1]
        self._glsl.upperCorner.z = value[2]

    def transform(self, trafo: Transform) -> RectBBox:
        """
        Returns new boundary box encompassing this one after applying the given
        transformation to it.
        """
        # apply trafo to all corners and use them to get new one
        x = [self.upperCorner[0], self.lowerCorner[0]]
        y = [self.upperCorner[1], self.lowerCorner[1]]
        z = [self.upperCorner[2], self.lowerCorner[2]]
        corners = np.array(list(itertools.product(x, y, z)))
        corners = trafo.apply(corners)
        return RectBBox(tuple(corners.min(0)), tuple(corners.max(0)))


class SphereBBox:
    """
    Spherical bounding box defined by its center and radius.
    Corresponds to a single point if radius is zero.

    Parameters
    ----------
    center: tuple[float, float, float]
        center of the sphere
    radius: float
        radius of the sphere
    """

    class GLSL(Structure):
        """GLSL struct equivalent"""

        _fields_ = [("center", vec3), ("radius", c_float)]

    def __init__(self, center: tuple[float, float, float], radius: float) -> None:
        self._glsl = self.GLSL()
        self.center = center
        self.radius = radius

    @property
    def glsl(self) -> SphereBBox.GLSL:
        """The underlying GLSL structure to be consumed by shaders"""
        return self._glsl

    @property
    def center(self) -> tuple[float, float, float]:
        """Center of the sphere"""
        return (
            self._glsl.center.x,
            self._glsl.center.y,
            self._glsl.center.z,
        )

    @center.setter
    def center(self, value: tuple[float, float, float]) -> None:
        self._glsl.center.x = value[0]
        self._glsl.center.y = value[1]
        self._glsl.center.z = value[2]

    @property
    def radius(self) -> float:
        """Radius of the sphere"""
        return self._glsl.radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._glsl.radius = value


def _createMeshFromTrimesh(mesh: trimesh.Trimesh) -> hp.Mesh:
    """util function converting a mesh from trimesh to hephaistos"""
    result = hp.Mesh()
    vertices = np.concatenate((mesh.vertices, mesh.vertex_normals), axis=-1)
    result.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    result.indices = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    return result


def loadMesh(filepath: str | Path) -> hp.Mesh:
    """
    Loads the mesh stored at the given file path and returns a
    `hephaistos.Mesh` to be used in MeshStore.
    """
    mesh = trimesh.load_mesh(filepath)
    return _createMeshFromTrimesh(mesh)


class MeshInstance:
    """
    Instance of a mesh by referencing the corresponding one stored in a
    MeshStore. It can also be assigned a material by name, which will get
    resolved during compilation of the scene.
    """

    def __init__(
        self,
        instance: hp.GeometryInstance,
        vertices: int,
        indices: int,
        triangleCount: int,
        bbox: RectBBox,
        material: str,
        objectId: int | Sequence[int] = 0,
    ) -> None:
        self._instance = instance
        self._vertices = vertices
        self._indices = indices
        self._triangleCount = triangleCount
        self._localBbox = bbox
        self._bbox = self.localBBox.transform(self.transform)
        self.material = material
        self.objectId = objectId
        self.isPortal: bool = False
        self.portalMeshKey: str | None = None
        self.contextCount: int = 1
        # Portal transitions `(arrivalInstance, nextContext)` indexed by the CURRENT 
        # context the photon carries
        self.portalLinks: list[tuple["MeshInstance", int] | None] = []

    @property
    def bbox(self) -> RectBBox:
        """Rectangular boundary box encompassing the mesh after transformation"""
        return self._bbox

    @property
    def localBBox(self) -> RectBBox:
        """Rectangular boundary box encompassing the mesh before transformation."""
        return self._localBbox

    @property
    def instance(self) -> hp.GeometryInstance:
        """The underlying geometry instance"""
        return self._instance

    @property
    def vertices(self) -> int:
        """Device address on the gpu where the vertex data is stored"""
        return self._vertices

    @property
    def indices(self) -> int:
        """Device address on the gpu where the index data is stored"""
        return self._indices

    @property
    def material(self) -> str:
        """Name of the material this mesh instance consists of"""
        return self._material

    @material.setter
    def material(self, value: str) -> None:
        self._material = value

    @property
    def objectId(self) -> int | list[int]:
        """
        Object id reported when instance is used as detector/target. A list of
        ids assigns one id per portal context of the enclosing scene, see
        `objectIds`.
        """
        return self._objectId

    @objectId.setter
    def objectId(self, value: int | Sequence[int]) -> None:
        ids: int | list[int]
        # check for the scalar case first: str is a Sequence, ndarray is not
        if isinstance(value, (int, np.integer)):
            ids = int(value)
        else:
            ids = [int(i) for i in value]
            if len(ids) == 0:
                raise ValueError("objectId must not be an empty sequence!")
        self._objectId = ids

    @property
    def objectIds(self) -> list[int]:
        """
        Object ids indexed by the portal context the photon
        carries, always as a list. A scalar `objectId` yields a single element
        and is broadcast to all contexts by the tracer.
        """
        if isinstance(self._objectId, list):
            return list(self._objectId)
        return [self._objectId]

    @property
    def triangleCount(self) -> int:
        """Amount of triangles the referenced Mesh consists of"""
        return self._triangleCount

    @property
    def transform(self) -> Transform:
        """
        The 3x4 transformation matrix applied on the underlying mesh to
        create this instance
        """
        return Transform(self.instance.transform)

    @transform.setter
    def transform(self, value: ArrayLike) -> None:
        self.instance.transform = value
        self._bbox = self.localBBox.transform(self.transform)


class MeshStore:
    """
    Class managing the lifetime of single meshes allowing to reuse them.
    """

    def __init__(self, meshes: Mapping[str, hp.Mesh | str]) -> None:
        """
        Creates a new MeshStore managing the lifetime of meshes.

        Parameters
        ----------
        meshes: mapping of named meshes (hephaistos.Mesh or filepath)
        """
        # load all meshes that are specified as file paths
        self._keys = list(meshes.keys())
        values = [loadMesh(v) if isinstance(v, str) else v for v in meshes.values()]
        self._triangleCounts = [len(mesh.indices) for mesh in values]
        _lower = [tuple(mesh.vertices[:, :3].min(0)) for mesh in values]
        _upper = [tuple(mesh.vertices[:, :3].max(0)) for mesh in values]
        self._bbox = [RectBBox(l, u) for l, u in zip(_lower, _upper)]
        # pass meshes to hephaistos to build blas
        self._store = hp.GeometryStore(values)

    def createInstance(
        self,
        key: str,
        material: str,
        transform: Transform | None = None,
        *,
        detectorId: int | Sequence[int] = 0,
        scale: float | None = None,
    ) -> MeshInstance:
        """
        Creates and returns a new MeshInstance of a mesh specified via its name.
        Optionally, a material can be assigned to the new instance.

        Parameters
        ----------
        key: str
            Name of the mesh as specified during init of store
        material: Optional[str], default = None
            Name of the assigned material. The actual material will get resolved
            during compilation of the scene.
        transform: Optional[Transform], default = None
            The transformation to apply on the instance.
            If None, identity transformation is applied.
        detectorId: int | Sequence[int], default = 0
            Id of the instance if used as a detector/target. A sequence assigns
            one id per portal context of the scene the instance is placed in,
            allowing the placements of a shared sub-scene to be told apart. Only
            a tracer given a `MultiScene` evaluates the per-context ids; every
            other tracer uses the first one.
        scale: float | None, default=None
            Dimension of the vertex positions. Defaults to 1m.

        Returns
        -------
        instance: MeshInstance
            The new created instance
        """
        idx = self._keys.index(key)
        geo = self._store.geometries[idx]
        instance = MeshInstance(
            self._store.createInstance(idx),
            geo.vertices_address,
            geo.indices_address,
            self._triangleCounts[idx],
            self._bbox[idx],
            material,
            detectorId,
        )
        if scale is None:
            # default to 1m
            scale = 1.0 * u.m
        if scale != 1.0 or transform is not None:
            trafo = Transform.Scale(scale, scale, scale)
            if transform is not None:
                # scale first
                trafo = transform @ trafo
            instance.transform = trafo.numpy()
        return instance

    def createPortal(
        self,
        key: str,
        material: str,
        transform: Transform | None = None,
        *,
        contextCount: int = 1,
        scale: float | None = None,
    ) -> MeshInstance:
        """
        Creates a portal instance of a mesh. Placing the portal is separate from
        linking it: after creating all portals, call `linkPortals` to define
        where a crossing continues. The surface model runs normally and a
        TRANSMITTED ray switches to the arrival portal's scene/frame
        (reflected rays stay); see `MultiScene`.

        The material must carry the portal flag `"P"`. Any surface model can be 
        used. For a portal that should be physically inert, use a `BorderSurface` 
        with the SAME medium on both sides and flags `"P*"`.

        Parameters
        ----------
        key: str
            Name of the mesh as specified during init of store
        material: str
            Name of the assigned material. The actual material will get resolved
            during compilation of the scene. The material must carry the portal 
            flag `"P"`.
        transform: Transform | None, default=None
            The transformation to apply on the instance.
            If None, identity transformation is applied.
        contextCount: int, default=1
            Number of portal contexts of the scene this box will be placed in,
            i.e. how often that scene is entered through a portal. Must be the same
            for all portals of one scene.
        scale: float | None, default=None
            Dimension of the vertex positions. Defaults to 1m.
        """
        if contextCount < 1:
            raise ValueError("contextCount must be at least 1!")
        instance = self.createInstance(key, material, transform, scale=scale)
        instance.isPortal = True
        instance.portalMeshKey = key
        instance.contextCount = contextCount
        instance.portalLinks = [None] * contextCount
        return instance


def linkPortals(
    a: MeshInstance,
    b: MeshInstance,
    contexts: Iterable[tuple[int, int]],
) -> None:
    """
    Connects two portals so photons crossing one continue at the other.

    `contexts` lists `(contextA, contextB)` pairs: a photon crossing `a` while
    carrying context `contextA` arrives at `b` carrying `contextB`, and vice
    versa. Contexts of an instance may be linked in several calls.

    Parameters
    ----------
    a, b: MeshInstance
        The portal boxes to connect, as returned by `MeshStore.createPortal`.
        They must be instances of the same mesh.
    contexts: Iterable[(int, int)]
        Context pairs to connect, see above.
    """
    for inst, name in ((a, "a"), (b, "b")):
        if not inst.isPortal:
            raise ValueError(
                f"Instance {name} is not a portal (use MeshStore.createPortal)."
            )
    if a.portalMeshKey != b.portalMeshKey:
        raise ValueError(
            f"Linked portal boxes must share the same mesh "
            f"({a.portalMeshKey!r} -> {b.portalMeshKey!r})."
        )

    pairs = [(int(ctxA), int(ctxB)) for ctxA, ctxB in contexts]
    # validate everything before linking anything
    claimed: set[tuple[int, int]] = set()
    for ctxA, ctxB in pairs:
        slots = [(a, ctxA, "a"), (b, ctxB, "b")]
        if a is b and ctxA == ctxB:
            slots = slots[:1]  # a context linked to itself claims one slot
        for inst, ctx, name in slots:
            if not 0 <= ctx < inst.contextCount:
                raise ValueError(
                    f"Context {ctx} is out of range for portal {name}, which "
                    f"declares {inst.contextCount} context(s)."
                )
            if inst.portalLinks[ctx] is not None or (id(inst), ctx) in claimed:
                raise ValueError(
                    f"Context {ctx} of portal {name} is already linked."
                )
            claimed.add((id(inst), ctx))

    for ctxA, ctxB in pairs:
        a.portalLinks[ctxA] = (b, ctxB)
        b.portalLinks[ctxB] = (a, ctxA)


class SceneBase:
    """
    Common base of `Scene` and `MultiScene` holding everything a tracer needs
    from the geometry: the acceleration structure a ray starts out against, the
    boundary it is confined to, the materials it may encounter and the preamble
    macros the geometry requires.

    Parameters
    ----------
    materials: MaterialStore
        Store containing the material and media referenced by the tracer.
    bbox: RectBBox, default=None
        bounding box containing the scene, limiting traced rays inside. Defaults
        to a cube of 1km in each primal direction.
    """

    sbtHitStride: Final[int] = 2

    def __init__(
        self,
        materials: MaterialStore,
        bbox: RectBBox | None = None,
    ) -> None:
        self._materials = materials
        if bbox is None:
            bbox = RectBBox((-1.0, -1.0, -1.0) * u.km, (1.0, 1.0, 1.0) * u.km)
        self.bbox = bbox

    @classmethod
    def _buildTlas(
        cls,
        instances: list[MeshInstance],
        materials: MaterialStore,
    ) -> hp.AccelerationStructure:
        """
        Fills in the data the shader resolves per instance and builds the
        corresponding acceleration structure. That data is namely:
         - customIndex storing the index into the material table
         - sbtOffset mapping to surface model index
        Note, that we will have two hit shaders per surface model (one for
        tracing, one for NEE), so the sbtOffset is actually twice the index.
        """
        if len(instances) == 0:
            raise ValueError("No instances given. A scene cannot be empty!")
        geomInstances: list[hp.GeometryInstance] = []
        for inst in instances:
            if inst.material not in materials:
                raise ValueError(f'Unknown material "{inst.material}"')
            matIdx = materials[inst.material]
            srfIdx = materials.surfaceModelMap[matIdx]

            geom = inst.instance
            geom.customIndex = matIdx
            geom.instanceSBTOffset = cls.sbtHitStride * srfIdx

            geomInstances.append(geom)
        return hp.AccelerationStructure(geomInstances)

    @property
    def bbox(self) -> RectBBox:
        """The bounding box containing the scene, limiting traced rays inside"""
        return self._bbox

    @bbox.setter
    def bbox(self, value: RectBBox) -> None:
        self._bbox = value

    @property
    def macros(self) -> dict[str, bool]:
        """Preamble macros a tracer must define to handle this geometry"""
        return {}

    @property
    def materials(self) -> MaterialStore:
        """Store containing the materials and media used in the scene."""
        return self._materials

    @property
    def tlas(self) -> hp.AccelerationStructure:
        """The acceleration structure describing the scene's geometry"""
        raise NotImplementedError

    def bindParams(self, program: hp.Program | hp.RayTracingPipeline) -> None:
        self.materials.bindParams(program)


class Scene(SceneBase):
    """
    A scene describes a structure that the shader can query rays against and
    retrieve data about hit geometries like vertices and material.

    Parameters
    ----------
    instances: Iterable[MeshInstance]
        instances that make up the scene
    materials: MaterialStore
        Store containing the material and media referenced by the tracer.
    bbox: RectBBox, default=None
        bounding box containing the scene, limiting traced rays inside. Defaults
        to a cube of 1km in each primal direction.

    See Also
    --------
    MultiScene: geometry split into sub-scenes connected by portals
    """

    def __init__(
        self,
        instances: Iterable[MeshInstance],
        materials: MaterialStore,
        *,
        bbox: RectBBox | None = None,
    ) -> None:
        super().__init__(materials, bbox)
        self._instances = list(instances)
        self._tlas = self._buildTlas(self._instances, materials)
        # a single scene has no portal contexts, so the first id is the only one
        objectIdMap = [inst.objectIds[0] for inst in self._instances]
        self._objectIdMap = hp.Tensor(
            np.ascontiguousarray(objectIdMap, dtype=np.int32)
        )

    @property
    def instances(self) -> list[MeshInstance]:
        """Mesh instances making up the scene, in gl_InstanceID order."""
        return self._instances

    @property
    def objectIdMapTensor(self) -> hp.Tensor:
        """Tensor containing the mapping from instance id to object id"""
        return self._objectIdMap

    @property
    def tlas(self) -> hp.AccelerationStructure:
        """The acceleration structure describing the scene's geometry"""
        return self._tlas

    def bindParams(self, program: hp.Program | hp.RayTracingPipeline) -> None:
        super().bindParams(program)
        program.bindParams(ObjectIdMap=self.objectIdMapTensor)


class MultiScene(SceneBase):
    """
    Geometry split into several sub-scenes, each traced in its own LOCAL
    coordinate frame. This can be used to trace fine geometries where single 
    precision world coordinates offer too low resolution.

    Sub-scenes are entered through portals. Place portals with 
    `MeshStore.createPortal` and connect them with `linkPortals` BEFORE 
    initializing the MultiScene, which then validates the complete transition 
    graph.

    `scenes[0]` is the world the rays start out in, the remaining entries are
    the sub-scenes. The same sub-scene can be placed multiple times, in this
    case the `context` can be used to distinguish between the different 
    instances. It selects both where a crossing through a portal continues
    and, via a per-context `detectorId`, which detector a hit is reported as.

    Parameters
    ----------
    scenes: Iterable[Iterable[MeshInstance]]
        Instances of each scene, world scene first. The order within a scene
        defines the instance ids the transition tables are built against.
    materials: MaterialStore
        Store containing the material and media referenced by the tracer. Shared
        by all scenes, which also makes their surface model indices - and hence
        the single shader binding table - consistent by construction.
    bbox: RectBBox, default=None
        Bounding box of the WORLD scene, limiting traced rays inside. Defaults
        to a cube of 1km in each primal direction.

    See Also
    --------
    theia.trace.SceneForwardTracer: the tracer consuming this
    """

    def __init__(
        self,
        scenes: Iterable[Iterable[MeshInstance]],
        materials: MaterialStore,
        *,
        bbox: RectBBox | None = None,
    ) -> None:
        super().__init__(materials, bbox)
        # materialize eagerly: the order defines the instance ids the tables are
        # built against, and the argument may well be a one-shot generator
        self._scenes = [list(insts) for insts in scenes]
        if len(self._scenes) < 2:
            raise ValueError(
                "A MultiScene needs the world scene plus at least one sub-scene; "
                "use Scene for a single one."
            )
        # an instance carries the transform of exactly one placement, and its
        # identity is what the transition graph is resolved against
        seen: dict[int, int] = {}
        for sid, insts in enumerate(self._scenes):
            for inst in insts:
                if id(inst) in seen:
                    raise ValueError(
                        f"Instance appears in both scene {seen[id(inst)]} and "
                        f"scene {sid}. Create a separate instance per placement."
                    )
                seen[id(inst)] = sid
        self._loc = {
            id(inst): (sid, iid)
            for sid, insts in enumerate(self._scenes)
            for iid, inst in enumerate(insts)
        }

        self._tlasList = [self._buildTlas(insts, materials) for insts in self._scenes]
        self._contextCount = self._validatePortals()
        self._buildTables()

    @property
    def contextCounts(self) -> list[int]:
        """Number of portal contexts per scene, indexed by scene id."""
        return list(self._contextCount)

    @property
    def macros(self) -> dict[str, bool]:
        return {"RAY_PORTAL": True}

    @property
    def scenes(self) -> list[list[MeshInstance]]:
        """Instances of each scene, world scene first, in gl_InstanceID order."""
        return self._scenes

    @property
    def tlas(self) -> hp.AccelerationStructure:
        """Acceleration structure of the world scene, where rays start out"""
        return self._tlasList[0]

    @property
    def subTlas(self) -> list[hp.AccelerationStructure]:
        """Acceleration structures of the sub-scenes, indexed by (sceneId - 1)"""
        return self._tlasList[1:]

    def bindParams(self, program: hp.Program | hp.RayTracingPipeline) -> None:
        super().bindParams(program)
        program.bindParams(**self._tensors)

    @staticmethod
    def _splitAddr(address: int) -> tuple[int, int]:
        """Split a 64-bit device address into (lo, hi) uint32 (a uvec2)."""
        return (address & 0xFFFFFFFF, (address >> 32) & 0xFFFFFFFF)

    def _validatePortals(self) -> list[int]:
        """
        Validates the portal transition graph and raises a clear error on misuse:
          - a portal's material MUST carry the PORTAL flag on both sides (and
            vice versa) so the shader's MATERIAL_PORTAL_BIT gate matches the
            transition tables,
          - the world scene defines at least one portal,
          - all portals within a scene declare the same number of contexts,
          - every context of every portal is linked, and targets a portal with a
            matching mesh key (a closed, total transition graph),
          - every link is bidirectional, i.e. crossing the arrival portal back
            returns to the source portal and context,
          - the net cross-portal map M = T_arrival . T_source^-1 is a rigid
            isometry (matched scale on both boxes),
          - per-context detector ids match the context count of their scene.

        Returns the per-scene context count, indexed by scene id.
        """
        matTable = self._materials.materials  # PropertyTable of the MaterialStore

        def hasPortalFlag(name: str) -> bool:
            # read the per-direction flags from the compiled material table
            entry = matTable.entries[name]
            return bool(entry["inwards"].flags & MaterialFlags.PORTAL) and bool(
                entry["outwards"].flags & MaterialFlags.PORTAL
            )

        # portals per scene + per-scene context count. Enforce isPortal <-> PORTAL
        # flag, so a flagged surface that is not a portal (or vice versa) is
        # caught here rather than misbehaving on the device.
        portalsPerScene: list[list[MeshInstance]] = []
        contextCount: list[int] = []
        for sid, insts in enumerate(self._scenes):
            portals = []
            for inst in insts:
                hasFlag = hasPortalFlag(inst.material)
                if inst.isPortal != hasFlag:
                    raise ValueError(
                        f"Inconsistent portal in scene {sid}: instance material "
                        f"{inst.material!r} "
                        + (
                            "was created as a portal but its material lacks the "
                            "PORTAL flag on both sides"
                            if inst.isPortal
                            else "carries the PORTAL flag but was not created "
                            "with MeshStore.createPortal"
                        )
                        + " - a portal needs both."
                    )
                if inst.isPortal:
                    portals.append(inst)
            portalsPerScene.append(portals)
            counts = {p.contextCount for p in portals}
            if len(counts) > 1:
                raise ValueError(
                    f"Portals in scene {sid} declare differing context counts "
                    f"{sorted(counts)}; the context count is a property of the "
                    "scene, so every portal of it must agree."
                )
            contextCount.append(counts.pop() if portals else 0)
        if len(portalsPerScene[0]) == 0:
            raise ValueError(
                "The world scene defines no portals, so no sub-scene can ever "
                "be entered (use MeshStore.createPortal)."
            )
        for sid, portals in enumerate(portalsPerScene[1:], start=1):
            if len(portals) == 0:
                raise ValueError(
                    f"Sub-scene {sid} defines no portals."
                )

        # validate every link: fully linked, target exists, mesh key, rigid map
        eye3 = np.eye(3)
        for sid, portals in enumerate(portalsPerScene):
            for p in portals:
                Lsrc = p.transform.numpy()[:, :3]
                try:
                    LsrcInv = np.linalg.inv(Lsrc)
                except np.linalg.LinAlgError:
                    raise ValueError(f"A portal transform in scene {sid} is singular.")
                for ctx, link in enumerate(p.portalLinks):
                    if link is None:
                        raise ValueError(
                            f"Context {ctx} of a portal in scene {sid} is not "
                            "linked. Every context needs a transition."
                        )
                    arrival, nextCtx = link
                    if id(arrival) not in self._loc:
                        raise ValueError(
                            f"Portal link in scene {sid} (context {ctx}) targets "
                            "an instance that is not part of this MultiScene."
                        )
                    dstScene, _ = self._loc[id(arrival)]
                    if not 0 <= nextCtx < contextCount[dstScene]:
                        raise ValueError(
                            f"Portal link in scene {sid} (context {ctx}) targets "
                            f"context {nextCtx} of scene {dstScene}, which has "
                            f"{contextCount[dstScene]} context(s)."
                        )
                    if p.portalMeshKey != arrival.portalMeshKey:
                        raise ValueError(
                            f"Linked portal boxes must share the same mesh "
                            f"({p.portalMeshKey!r} -> {arrival.portalMeshKey!r})."
                        )
                    back = arrival.portalLinks[nextCtx]
                    if back is None or back[0] is not p or back[1] != ctx:
                        raise ValueError(
                            f"Portal link in scene {sid} (context {ctx}) is "
                            "one-way: crossing the arrival portal back does not "
                            "return to it. Links must be bidirectional (use "
                            "linkPortals instead of editing portalLinks)."
                        )
                    Larr = arrival.transform.numpy()[:, :3]
                    Mlin = Larr @ LsrcInv
                    if not np.allclose(Mlin @ Mlin.T, eye3, atol=1e-4):
                        raise ValueError(
                            f"Portal link in scene {sid} (context {ctx}) has a "
                            "non-rigid net map: source and arrival box scales "
                            "differ. Linked boxes must have matching scale."
                        )

        # per-context detector ids must cover exactly the scene's contexts. A
        # scalar id is broadcast and therefore always fine.
        for sid, insts in enumerate(self._scenes):
            nCtx = max(contextCount[sid], 1)
            for iid, inst in enumerate(insts):
                ids = inst.objectIds
                if len(ids) != 1 and len(ids) != nCtx:
                    raise ValueError(
                        f"Instance {iid} of scene {sid} assigns {len(ids)} "
                        f"detector ids, but the scene has {nCtx} portal "
                        "context(s). Pass one id per context, or a single id "
                        "to use for all of them."
                    )

        return contextCount

    def _buildTables(self) -> None:
        """
        Builds the SSBOs read by portal.glsl:
          - SubTlasTable: sub-scene TLAS device addresses, indexed by
            (sceneId - 1) (world excluded, it uses params.tlas);
          - TransitionTable / ObjectIdTable: per-scene (indexed by sceneId,
            world at 0) device addresses of the flattened transition table
            (14 uint words per record: nextScene, nextContext, then a mat4x3 as
            12 column-major floats) and the per-context detector ids. Both are
            [context][instance] tables addressed by gl_InstanceID directly;
            non-portal instances hold a zero transition record;
          - InstanceCountTable: per-scene instance count, i.e. the row stride of
            these tables.
        """
        subTlas = np.array(
            [self._splitAddr(t.address) for t in self.subTlas], dtype=np.uint32
        ).reshape(-1)

        keep: list = []  # keep the per-scene tensors alive (their addresses are used)
        transAdr: list = []
        objIdAdr: list = []
        instanceCount: list[int] = []
        for sid, insts in enumerate(self._scenes):
            # transition table: rows = contexts, cols = instances, 14 words
            # each. Spending a record on every instance instead of only on the
            # portals avoids a dependent gl_InstanceID -> portal index lookup.
            nCtx = self._contextCount[sid]
            nInst = len(insts)
            records = np.zeros((max(nCtx * nInst, 1), 14), dtype=np.uint32)
            for iid, p in enumerate(insts):
                if not p.isPortal:
                    continue
                for ctx, link in enumerate(p.portalLinks):
                    assert link is not None  # guaranteed by _validatePortals
                    arrival, nextCtx = link
                    dstScene, _ = self._loc[id(arrival)]
                    A = arrival.transform.numpy()  # (3,4) float32
                    T = np.ascontiguousarray(A.T, dtype=np.float32).reshape(-1)
                    r = ctx * nInst + iid
                    records[r, 0] = np.uint32(dstScene)
                    records[r, 1] = np.uint32(nextCtx)
                    records[r, 2:14] = T.view(np.uint32)
            transT = hp.Tensor(np.ascontiguousarray(records.reshape(-1)))
            keep.append(transT)
            transAdr.append(self._splitAddr(transT.address))

            # objectId table: rows = contexts, cols = instances. Scalar ids get
            # broadcast over all contexts.
            objIds = np.empty((max(nCtx, 1), nInst), dtype=np.int32)
            for iid, inst in enumerate(insts):
                objIds[:, iid] = inst.objectIds  # broadcasts a single id
            objT = hp.Tensor(np.ascontiguousarray(objIds.reshape(-1)))
            keep.append(objT)
            objIdAdr.append(self._splitAddr(objT.address))

            instanceCount.append(nInst)

        toTensor = lambda rows, dtype: hp.Tensor(
            np.ascontiguousarray(np.array(rows, dtype=dtype).reshape(-1))
        )
        self._keepAlive = keep
        self._tensors = {
            "SubTlasTable": hp.Tensor(np.ascontiguousarray(subTlas)),
            "TransitionTable": toTensor(transAdr, np.uint32),
            "ObjectIdTable": toTensor(objIdAdr, np.uint32),
            "InstanceCountTable": toTensor(instanceCount, np.uint32),
        }


class SceneTemplate:
    """
    Template for creating scenes. Loads meshes and partial scenes from a file
    and provides methods to create complete scenes.

    Parameters
    ----------
    file: str | Path
        Path to the file containing the scene to be loaded
    materials: MaterialStore | Mapping[str, int] | None, default=None
        Material map used when creating scenes from this template. If None, it
        must be explicitly passed to `createScene`.
    templateTransform: Transform | None, default=None
        Optional transformation to be applied to the loaded template
    detectorIdMap: Mapping[str, int] | None, default=None
        Optional map of instances to their detectorId. Instances not mapped will
        receive an id of 0. If None, each instance gets a unique id starting
        from 1 counting up.
    detectorMaterial: set[str] | None, default=None
        If provided, only meshes with material contained in this list will get a
        unique detector id. The remaining ones will all get a detector id of 0.
        Ignored, if `detectorIdMap` is provided.
    """

    @dataclass(frozen=True)
    class InstanceInfo:
        name: str
        """Instance name"""
        meshName: str
        """Name of the referenced mesh"""
        transform: Transform
        """Base transformation from mesh to template"""
        material: str
        """Name of referenced material"""
        detectorId: int
        """Detector id assigned to instance when creating scene"""

    def __init__(
        self,
        file: str | Path,
        *,
        materials: MaterialStore | Mapping[str, int] | None = None,
        sceneMedium: int = 0,
        templateTransform: Transform | None = None,
        detectorIdMap: Mapping[str, int] | None = None,
        detectorMaterial: set[str] | None = None,
    ) -> None:
        # load file
        scene: trimesh.Scene = trimesh.load(file, force="scene")

        def getMaterialName(meshName: str, mesh: trimesh.Trimesh) -> str:
            # to avoid a lengthy check for types and Nones at multiple occasion,
            # we simply put the member access inside a try except block
            name = ""
            try:
                name = mesh.visual.material.name
            except AttributeError:
                # do nothing for now, we handle it in the following
                pass

            # check if we got a meaningful name
            # second check is for default name of trimesh
            if name == "" or name == "material_0":
                raise ValueError(f'Mesh "{meshName}" has no material assigned!')

            # done
            return name

        # collect material mapped (defined per mesh)
        matDict = {name: getMaterialName(name, m) for name, m in scene.geometry.items()}
        # load meshes to GPU
        meshes = {name: _createMeshFromTrimesh(m) for name, m in scene.geometry.items()}
        self._store = MeshStore(meshes)

        # assemble instances
        instances: dict[str, SceneTemplate.InstanceInfo] = {}
        nextId = 1
        base_frame = scene.graph.base_frame
        for instanceName in scene.graph.nodes_geometry:
            # fetch trafo and mesh
            trafo, meshName = scene.graph.get(instanceName, base_frame)
            trafo = Transform(trafo[:3, :])
            if templateTransform is not None:
                trafo @= templateTransform
            trafo.freeze()
            # get material name
            mat = matDict[meshName]
            # calculate detectorId
            detId = 0
            if detectorIdMap is not None:
                detId = detectorIdMap.get(instanceName, 0)
            elif detectorMaterial is not None:
                if mat in detectorMaterial:
                    detId = nextId
                    nextId += 1
            else:
                detId = nextId
                nextId += 1

            # create instance
            instance = self.InstanceInfo(instanceName, meshName, trafo, mat, detId)
            instances[instanceName] = instance

        self._instances = MappingProxyType(instances)
        self.materials = materials
        self._sceneMedium = sceneMedium
        self._idStride = nextId - 1

    @property
    def materials(self) -> Mapping[str, int] | None:
        """
        Material map used when creating scenes from this template. If None, it
        must be explicitly passed to `createScene`.
        """
        return self._materials

    @materials.setter
    def materials(self, value: MaterialStore | Mapping[str, int] | None) -> None:
        if isinstance(value, MaterialStore):
            value = value.material
        self._materials = value

    @property
    def meshStore(self) -> MeshStore:
        """MeshStore containing the meshes used by this template"""
        return self._store

    @property
    def instances(self) -> MappingProxyType[str, InstanceInfo]:
        return self._instances

    @property
    def sceneMedium(self) -> int:
        """Device address of the medium the template is embedded within"""
        return self._sceneMedium

    @sceneMedium.setter
    def sceneMedium(self, value: int) -> None:
        self._sceneMedium = value

    def createScene(
        self,
        tempInstance: Iterable[Transform | None] | None = None,
        *,
        materials: MaterialStore | Mapping[str, int] | None = None,
        sceneMedium: int = 0,
        sceneTransformation: Transform | None = None,
        sceneBBox: RectBBox | None = None,
        detectorIdStride: int | None = None,
    ) -> tuple[Scene, dict[tuple[str, int], int]]:
        """
        Creates a new scene consisting of copies of this template for each given
        transformation, which is applied to the corresponding copy.

        Parameters
        ----------
        tempInstance: Iterable[Transform] | None, default=None
            List of `Transform` for each template instance. If None, a single
            instance without a transformation is used.
        materials: MaterialStore | Mapping[str, int] | None, default=None
            Material map used when creating scenes from this template. If None,
            uses the one passed to the template.
        sceneMedium: int, default=0
            Device address of the medium the scene is emerged in, e.g. the
            address of a water medium for an underwater simulation. Defaults to
            zero specifying vacuum.
        sceneTransformation: Transform | None, default=None
            Optional transformation applied to the loaded scene
        sceneBBox : RectBBox | None, default=None
            Bounding box containing the scene, limiting traced rays inside.
            Defaults to a cube of 1km in each primal direction.
        detectorIdStride: int | None, default=None
            Offset applied to the detectorId of each instance for each template
            instance. If None, uses smallest possible stride.

        Returns
        -------
        scene: Scene
            The created scene
        detectorId: dict[tuple[str, int], int]
            Mapping from pair of instance name and template instance count to
            their detectorId.
        """
        # check material mapping
        if materials is None:
            materials = self.materials
        if materials is None:
            raise ValueError("No material mapping was provided!")
        if isinstance(materials, MaterialStore):
            materials = materials.material
        # single instance
        if tempInstance is None:
            tempInstance = [None]
        # stride
        if detectorIdStride is None:
            detectorIdStride = self._idStride

        # create all instances
        sceneInst = []
        detIdMap: dict[tuple[str, int], int] = {}
        for i, trafo in enumerate(tempInstance):
            offset = i * detectorIdStride
            for tempInst in self.instances.values():
                # calculate detector id
                id = tempInst.detectorId
                if id != 0:
                    id += offset
                    detIdMap[(tempInst.name, i)] = id
                # assemble transformation matrix
                t = tempInst.transform.copy()
                if trafo is not None:
                    t @= trafo
                if sceneTransformation is not None:
                    t @= sceneTransformation
                # create instance
                name = tempInst.meshName
                mat = tempInst.material
                inst = self.meshStore.createInstance(name, mat, t, detectorId=id)
                sceneInst.append(inst)

        # create scene
        scene = Scene(sceneInst, materials, medium=sceneMedium, bbox=sceneBBox)
        return scene, detIdMap
