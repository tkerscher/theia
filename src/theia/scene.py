from __future__ import annotations
import hephaistos as hp
import numpy as np
import importlib.resources
import os.path
import trimesh
from ctypes import Structure, c_float, c_uint64
from hephaistos.glsl import vec2, vec3
from numpy.typing import NDArray, ArrayLike
from typing import Iterable, Optional, Union


class Transform:
    """Util class for creating transformation matrices"""

    def __init__(self) -> None:
        self._arr = np.identity(4)

    def numpy(self) -> NDArray:
        """
        Returns a numpy array in the correct format to be used with mesh
        instances
        """
        return np.ascontiguousarray(self._arr[:3, :], dtype=np.float32)

    @staticmethod
    def Rotation(dx: float, dy: float, dz: float, angle: float) -> Transform:
        """
        Returns a rotation transformation around the given axis (dx,dy,dz) for
        angle radians.
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
        res._arr[:3, :3] += np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        return res

    def rotate(self, dx: float, dy: float, dz: float, angle: float) -> Transform:
        """
        Rotates the transformation around (dx,dy,dz) counter-clockwise angle
        radians.
        """
        self @= Transform.Rotation(dx, dy, dz, angle)
        return self

    @staticmethod
    def Scale(x: float, y: float, z: float) -> Transform:
        """Returns a scale transformation"""
        res = Transform()
        res._arr[0, 0] = x
        res._arr[1, 1] = y
        res._arr[2, 2] = z
        return res

    def scale(self, x: float, y: float, z: float) -> Transform:
        """Scales the transformation in each dimension independently"""
        self @= Transform.Scale(x, y, z)
        return self

    @staticmethod
    def Translation(x: float, y: float, z: float) -> Transform:
        """Returns a translation transformation"""
        res = Transform()
        res._arr[:3, -1] = (x, y, z)
        return res

    def translate(self, x: float, y: float, z: float) -> Transform:
        """Translates the transform by the given amount"""
        self @= Transform.Translation(x, y, z)
        return self

    def __matmul__(self, other: Transform) -> Transform:
        if type(other) != Transform:
            raise TypeError(other)
        result = Transform()
        result._arr = self._arr @ other._arr
        return result

    def __imatmul__(self, other: Transform) -> Transform:
        if type(other) != Transform:
            raise TypeError(other)
        self._arr = self._arr @ other._arr
        return self


def loadMesh(filepath: str) -> hp.Mesh:
    """
    Loads the mesh stored at the given file path and returns a
    hephaistos.Mesh to be used in MeshStore.
    """
    result = hp.Mesh()
    mesh = trimesh.load_mesh(filepath)
    vertices = np.concatenate((mesh.vertices, mesh.vertex_normals), axis=-1)
    result.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    result.indices = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    return result


class MeshInstance(hp.GeometryInstance):
    """ "
    Instance of a mesh by referencing the corresponding one stored in a
    MeshStore. It can also be assigned a material by name, which will get
    resolved during compilation of the scene.
    """

    def __init__(
        self,
        instance: hp.GeometryInstance,
        vertices: int,
        indices: int,
        material: Union[str, None],
    ) -> None:
        self._instance = instance
        self._vertices = vertices
        self._indices = indices
        self.material = material

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
    def transform(self) -> NDArray[np.float32]:
        """
        The 3x4 transformation matrix applied on the underlying mesh to
        create this instance
        """
        return self.instance.transform

    @transform.setter
    def transform(self, value: ArrayLike) -> None:
        self.instance.transform = value


class MeshStore:
    """
    Class managing the lifetime of single meshes allowing to reuse them.
    """

    def __init__(self, meshes: dict[str, Union[hp.Mesh, str]]) -> None:
        """
        Creates a new MeshStore managing the lifetime of meshes.

        Parameters
        ----------
        meshes: dict of named meshes (hephaistos.Mesh or filepath)
        """
        # load all meshes that are specified as file paths
        self._keys = list(meshes.keys())
        values = [loadMesh(v) if type(v) == str else v for v in meshes.values()]
        # pass meshes to hephaistos to build blas
        self._store = hp.GeometryStore(values)

    def createInstance(
        self,
        key: str,
        material: Union[str, None] = None,
        *,
        transform: Union[Transform, None] = None
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
            material,
        )
        if transform is not None:
            instance.transform = transform.numpy()
        return instance


class Scene:
    """
    A scene describes a structure that the shader can query rays against and
    retrieve data about hit geometries like vertices and material.
    """

    class GLSLGeometry(Structure):
        """Equivalent structure for the Geometry type used in the shader"""

        _fields_ = [
            ("vertices", c_uint64),
            ("indices", c_uint64),
            ("material", c_uint64),
        ]

    class GLSLScene(Structure):
        """Equivalent structure for the Scene type used in the shader"""

        _fields_ = [
            ("geometries", c_uint64),
            ("medium", c_uint64),
            # ("maxRayLength", c_float)
        ]

    def __init__(
        self, instances: Iterable[MeshInstance], materials: dict[str, int], medium: int
    ) -> None:
        """
        Creates a new scene using the given instances

        Parameters
        ----------
        instances: Iterable[MeshInstance]
            instances that make up the scene
        materials: dict[str, int]
            dictionary mapping material names to device addresses as obtained
            from bakeMaterials()
        medium: int
            device address of the medium the scene is emerged in, e.g. the
            address of a water medium for an underwater simulation
        """
        instances = list(instances)
        # collect geometries
        geometries = hp.ArrayBuffer(Scene.GLSLGeometry, len(instances))
        for i, inst in enumerate(instances):
            geometries[i].vertices = inst.vertices
            geometries[i].indices = inst.indices
            if inst.material is not None:
                geometries[i].material = materials.get(inst.material)
            else:
                geometries[i].material = 0
        # upload geometries to gpu
        self._geometries = hp.ArrayTensor(Scene.GLSLGeometry, len(instances))
        hp.execute(hp.updateTensor(geometries, self._geometries))

        # build scene ubo
        sceneBuffer = hp.StructureBuffer(Scene.GLSLScene)
        sceneBuffer.medium = medium
        sceneBuffer.geometries = self._geometries.address
        self._scene = hp.StructureTensor(Scene.GLSLScene)
        hp.execute(hp.updateTensor(sceneBuffer, self._scene))

        # in order to pass to hephaistos.AccelerationStructure, we need to
        # extract the hephaistos.GeometryInstance from the MeshInstance
        self._tlas = hp.AccelerationStructure([i.instance for i in instances])

    @property
    def scene(self) -> hp.ByteTensor:
        """The tensor holding the scene UBO"""
        return self._scene

    @property
    def tlas(self) -> hp.AccelerationStructure:
        """The acceleration structure describing the scene's geometry"""
        return self._tlas


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
        maxDistance: float = 100.0
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
        dimension: Optional[tuple[float, float]] = None,
        position: Optional[tuple[float, float, float]] = None,
        direction: Optional[tuple[float, float, float]] = None,
        up: Optional[tuple[float, float, float]] = None,
        maxDistance: Optional[float] = None
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
        self._program.bindParams(
            tlas=scene.tlas, Scene=scene.scene, outImage=self._image
        )
        # run shader
        hp.beginSequence().And(
            self._program.dispatchPush(
                bytes(self._glsl), self.width // 4, self.height // 4
            )
        ).Then(hp.retrieveImage(self._image, self._buffer)).Submit().wait()
        # return image
        return self._buffer.numpy()
