from ctypes import *
from hephaistos.glsl import mat4x3, uvec2, vec2, vec3

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
        ("rngStream", c_uint32),
        ("rngCount", c_uint32),
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
