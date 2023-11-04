import hephaistos as hp

import theia.material

from ctypes import *
from hephaistos.glsl import mat4x3, uvec2, vec2, vec3
from numpy.ctypeslib import as_array


class QueueBuffer(hp.RawBuffer):
    def __init__(self, itemType: Structure, n: int):
        self._size = 4 + n * sizeof(itemType)
        super().__init__(self._size)
        self._adr = super().address
        self._arr = (itemType * n).from_address(self._adr + 4)
        self._count = c_uint32.from_address(self._adr)
        self.count = n

    def numpy(self):
        return as_array(self._arr)

    @property
    def count(self):
        return self._count.value

    @count.setter
    def count(self, value):
        self._count.value = value


class QueueTensor(hp.ByteTensor):
    def __init__(self, itemType: Structure, n: int):
        super().__init__(4 + sizeof(itemType) * n)


class WaterModel(
    theia.material.WaterBaseModel,
    theia.material.HenyeyGreensteinPhaseFunction,
    theia.material.MediumModel,
):
    def __init__(self) -> None:
        theia.material.WaterBaseModel.__init__(self, 5.0, 1000.0, 35.0)
        theia.material.HenyeyGreensteinPhaseFunction.__init__(self, 0.6)

    ModelName = "water"


N_PHOTONS = 4


class Photon(Structure):
    _fields_ = [
        ("wavelength", c_float),
        ("time", c_float),
        ("lin_c", c_float),
        ("log_c", c_float),
        # medium constants
        ("n", c_float),
        ("vg", c_float),
        ("mu_s", c_float),
        ("mu_e", c_float),
    ]


class PhotonHit(Structure):
    _fields_ = [
        ("wavelength", c_float),
        ("time", c_float),
        ("contribution", c_float),
    ]


class Ray(Structure):
    _fields_ = [
        ("position", vec3),
        ("direction", vec3),
        ("rngIdx", c_uint32),
        ("medium", uvec2),  # uint64
        ("photons", Photon * N_PHOTONS),
    ]


class RayHit(Structure):
    _fields_ = [
        ("position", vec3),
        ("direction", vec3),
        ("normal", vec3),
        ("hits", PhotonHit * N_PHOTONS),
    ]


class InitParams(Structure):
    _fields_ = [
        # ("medium", c_uint64),
        ("medium", uvec2),
        ("count", c_uint32),
        ("rngStride", c_uint32),
    ]


class IntersectionItem(Structure):
    _fields_ = [
        ("ray", Ray),
        ("geometryIdx", c_int32),
        ("customIdx", c_int32),
        ("triangleIdx", c_int32),
        ("barys", vec2),
        ("obj2World", mat4x3),
        ("world2Obj", mat4x3),
    ]


class ShadowRayItem(Structure):
    _fields_ = [
        ("ray", Ray),
        ("dist", c_float),
    ]


class VolumeScatterItem(Structure):
    _fields_ = [("ray", Ray), ("dist", c_float)]


class Detector(Structure):
    _fields_ = [("position", vec3), ("radius", c_float)]


class TraceParams(Structure):
    _fields_ = [
        ("targetIdx", c_uint32),
        ("scatterCoefficient", c_float),
        ("maxTime", c_float),
        ("lowerBBoxCorner", vec3),
        ("upperBBoxCorner", vec3),
    ]
