from __future__ import annotations
import importlib.resources

import numpy as np
import hephaistos as hp
from hephaistos.glsl import vec2, vec3

import itertools
import os.path
import trimesh
import trimesh.scene
import trimesh.visual

import theia.units as u
from theia.material import MaterialStore

from collections.abc import Iterable, Mapping
from ctypes import Structure, c_float, c_uint64
from dataclasses import dataclass
from numpy.typing import NDArray, ArrayLike
from pathlib import Path
from types import MappingProxyType


__all__ = [
    "loadMesh",
    "MeshInstance",
    "MeshStore",
    "RectBBox",
    "Scene",
    "SceneRender",
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

    def apply(self, points: NDArray) -> NDArray:
        """Applies the transformation to the given points of shape (N,3)"""
        return points @ self._arr[:3, :3].T + self._arr[:3, 3]

    def applyVec(self, vector: NDArray) -> NDArray:
        """
        Applies the transformation to the given vectors of shape (N,3).
        Similar to `apply`, but translation are ignored.
        """
        return vector @ self._arr[:3, :3].T

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
        Returns a rotation transformation around the given axis (dx,dy,dz) for
        angle degrees.
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
        angle = np.deg2rad(angle)
        res._arr[:3, :3] += np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        return res

    def rotate(self, dx: float, dy: float, dz: float, angle: float) -> Transform:
        """
        Returns a copy of the transformation rotated around (dx,dy,dz)
        counter-clockwise for angle degrees.
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
            the last the rotation angle in radians.
        translate: (float, float, float), default=(0.0, 0.0, 0.0)
            Translation given as an offset added.
        """
        result = Transform()
        if type(scale) is tuple:
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
    ) -> None:
        self._instance = instance
        self._vertices = vertices
        self._indices = indices
        self._triangleCount = triangleCount
        self._localBbox = bbox
        self._bbox = self.localBBox.transform(self.transform)
        self.material = material

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
        values = [loadMesh(v) if type(v) == str else v for v in meshes.values()]
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
        detectorId: int = 0,
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
        detectorId: int, default = 0
            Id of the instance if used as a detector/target.
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
        )
        instance.instance.customIndex = detectorId
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


class Scene:
    """
    A scene describes a structure that the shader can query rays against and
    retrieve data about hit geometries like vertices and material.

    Parameters
    ----------
    instances: Iterable[MeshInstance]
        instances that make up the scene
    materials: Mapping[str, int]
        Mapping from material names to device addresses
    medium: int, default=0
        device address of the medium the scene is emerged in, e.g. the address
        of a water medium for an underwater simulation. Defaults to zero
        specifying vacuum.
    bbox: RectBBox, default=None
        bounding box containing the scene, limiting traced rays inside. Defaults
        to a cube of 1km in each primal direction.
    """

    class GLSLGeometry(Structure):
        """Equivalent structure for the Geometry type used in the shader"""

        _fields_ = [
            ("vertices", c_uint64),
            ("indices", c_uint64),
            ("material", c_uint64),
        ]

    def __init__(
        self,
        instances: Iterable[MeshInstance],
        materials: MaterialStore | Mapping[str, int],
        *,
        medium: int = 0,
        bbox: RectBBox | None = None,
    ) -> None:
        instances = list(instances)
        if len(instances) == 0:
            raise ValueError("No instances given. Scene cannot be empty!")
        # fetch materials from store if needed
        if isinstance(materials, MaterialStore):
            materials = materials.material
        # collect geometries
        geometries = hp.ArrayBuffer(Scene.GLSLGeometry, len(instances))
        for i, inst in enumerate(instances):
            geometries[i].vertices = inst.vertices
            geometries[i].indices = inst.indices
            if inst.material is not None:
                if inst.material not in materials:
                    raise ValueError(f'Unknown material "{inst.material}"')
                geometries[i].material = materials.get(inst.material)
            else:
                geometries[i].material = 0
        # upload geometries to gpu
        self._geometries = hp.ArrayTensor(Scene.GLSLGeometry, len(instances))
        hp.execute(hp.updateTensor(geometries, self._geometries))

        # in order to pass to hephaistos.AccelerationStructure, we need to
        # extract the hephaistos.GeometryInstance from the MeshInstance
        self._tlas = hp.AccelerationStructure([i.instance for i in instances])

        # save medium
        self.medium = medium
        # save bbox
        if bbox is None:
            bbox = RectBBox((-1.0 * u.km,) * 3, (1.0 * u.km,) * 3)
        self.bbox = bbox

    @property
    def bbox(self) -> RectBBox:
        """The bounding box containing the scene, limiting traced rays inside"""
        return self._bbox

    @bbox.setter
    def bbox(self, value: RectBBox) -> None:
        self._bbox = value

    @property
    def geometries(self) -> hp.ByteTensor:
        """The tensor holding the array of geometries in the scene"""
        return self._geometries

    @property
    def medium(self) -> int:
        """
        device address of the medium the scene is emerged in, e.g. the address
        of a water medium for an underwater simulation.
        """
        return self._medium

    @medium.setter
    def medium(self, value: int) -> None:
        self._medium = value

    @property
    def tlas(self) -> hp.AccelerationStructure:
        """The acceleration structure describing the scene's geometry"""
        return self._tlas

    def bindParams(self, program: hp.Program) -> None:
        """Binds the parameters describing the scene in the given program"""
        program.bindParams(tlas=self.tlas, Geometries=self.geometries)


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
        instances: dict[str, self.InstanceInfo] = {}
        nextId = 1
        base_frame = scene.graph.base_frame
        for instanceName in scene.graph.nodes_geometry:
            # fetch trafo and mesh
            trafo, meshName = scene.graph.get(instanceName, base_frame)
            trafo = Transform(trafo[:3, :])
            if templateTransform is not None:
                trafo @= templateTransform
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
        tempInstance: Iterable[Transform] | None = None,
        *,
        materials: MaterialStore | Mapping[str, int] | None = None,
        sceneMedium: int = 0,
        sceneTransformation: Transform | None = None,
        sceneBBox: RectBBox | None = None,
        detectorIdStride: int = 0,
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
        sceneTransform: Transform | None, default=None
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
                t = tempInst.transform
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


class SceneRender:
    """
    Simple orthogonal ray tracer render for debugging the scene.
    Colors encode the normals.
    """

    class GLSLPush(Structure):
        _fields_ = [
            ("dimension", vec2),
            ("position", vec3),
            ("direction", vec3),
            ("up", vec3),
            ("maxDistance", c_float),
        ]

    def __init__(
        self,
        *,
        width: int = 1024,
        height: int = 1024,
        dimension: tuple[float, float] = (1.0, 1.0),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (0.0, 1.0, 0.0),
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        maxDistance: float = 100.0,
    ) -> None:
        """
        Creates a new scene renderer

        Parameters
        ----------
        width: int, default=1024
            width of the final rendered image
        height: int, default=1024
            height of the final rendered image
        dimension: (float, float), default=(1.0,1.0)
            dimension of the rendered image in scene units
        position: (float, float, float), default=(0.0,0.0,0.0)
            position of the camera
        direction: (float, float, float), default=(1.0,0.0,0.0)
            direction the camera points
        up: (float, float, float), default=(0.0,1.0,0.0)
            upwards direction of the camera
        maxDistance: float, default=100.0
            max ray length
        """
        self._glsl = SceneRender.GLSLPush()
        # save params
        self._width = width
        self._height = height
        self.dimension = dimension
        self.position = position
        self.direction = direction
        self.up = up
        self.maxDistance = maxDistance
        # create image
        self._buffer = hp.ImageBuffer(width, height)
        self._image = hp.Image(hp.ImageFormat.R8G8B8A8_UNORM, width, height)
        # build shader
        shader_dir = importlib.resources.files("theia").joinpath("shader")
        shader_path = os.path.join(shader_dir, "scene.render.glsl")
        source = None
        with open(shader_path, "r") as file:
            source = file.read()
        compiler = hp.Compiler()
        compiler.addIncludeDir(shader_dir)
        code = compiler.compile(source)
        self._program = hp.Program(code)

    @property
    def width(self) -> int:
        """Width of the final rendered image"""
        return self._width

    @property
    def height(self) -> int:
        """Height of the final rendered image"""
        return self._height

    @property
    def dimension(self) -> tuple[float, float]:
        """Dimension of the rendered image in scene units"""
        return (self._glsl.dimension.x, self._glsl.dimension.y)

    @dimension.setter
    def dimension(self, value: tuple[float, float]) -> None:
        self._glsl.dimension.x = value[0]
        self._glsl.dimension.y = value[1]

    @property
    def position(self) -> tuple[float, float, float]:
        """Position of the camera"""
        return (
            self._glsl.position.x,
            self._glsl.position.y,
            self._glsl.position.z,
        )

    @position.setter
    def position(self, value: tuple[float, float, float]) -> None:
        self._glsl.position.x = value[0]
        self._glsl.position.y = value[1]
        self._glsl.position.z = value[2]

    @property
    def direction(self) -> tuple[float, float, float]:
        """Direction the camera points"""
        return (
            self._glsl.direction.x,
            self._glsl.direction.y,
            self._glsl.direction.z,
        )

    @direction.setter
    def direction(self, value: tuple[float, float, float]) -> None:
        self._glsl.direction.x = value[0]
        self._glsl.direction.y = value[1]
        self._glsl.direction.z = value[2]

    @property
    def up(self) -> tuple[float, float, float]:
        """The up direction of the camera"""
        return (
            self._glsl.up.x,
            self._glsl.up.y,
            self._glsl.up.z,
        )

    @up.setter
    def up(self, value: tuple[float, float, float]) -> None:
        self._glsl.up.x = value[0]
        self._glsl.up.y = value[1]
        self._glsl.up.z = value[2]

    @property
    def maxDistance(self) -> float:
        """Maximum ray distance in scene units"""
        return self._glsl.maxDistance

    @maxDistance.setter
    def maxDistance(self, value: float) -> None:
        self._glsl.maxDistance = value

    def render(
        self,
        scene: Scene,
        *,
        dimension: tuple[float, float] | None = None,
        position: tuple[float, float, float] | None = None,
        direction: tuple[float, float, float] | None = None,
        up: tuple[float, float, float] | None = None,
        maxDistance: float | None = None,
    ) -> NDArray:
        """
        Renders the given scene and returns the image as numpy array of
        shape (width, height, 4), where the last dimension are the channels
        red, green, blue, alpha.

        Parameters
        ----------
        scene: Scene
            scene to be rendered
        dimension: (float, float)|None, default=None
            dimension of the rendered image in scene units. Use property if None
        position: (float, float, float)|None, default=None
            position of the camera. Use property if None
        direction: (float, float, float)|None, default=None
            direction the camera points. Use property if None
        up: (float, float, float)|None, default=None
            upwards direction of the camera. Use property if None
        maxDistance: float|None, default=None
            max ray length. Use property if None
        """
        # update non None params
        if dimension is not None:
            self.dimension = dimension
        if position is not None:
            self.position = position
        if direction is not None:
            self.direction = direction
        if up is not None:
            self.up = up
        if maxDistance is not None:
            self.maxDistance = maxDistance

        # bind parameters
        scene.bindParams(self._program)
        self._program.bindParams(outImage=self._image)
        # run shader
        hp.beginSequence().And(
            self._program.dispatchPush(
                bytes(self._glsl), self.width // 4, self.height // 4
            )
        ).Then(hp.retrieveImage(self._image, self._buffer)).Submit().wait()
        # return image
        return self._buffer.numpy()
