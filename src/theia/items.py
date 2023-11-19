from __future__ import annotations

from ctypes import Structure, c_float, c_int32, c_uint32
from typing import Type


def createHitQueueItem(nPhotons: int) -> Type[Structure]:
    class HitQueueItem(Structure):
        _fields_ = [
            ("position", c_float * 3),
            ("direction", c_float * 3),
            ("normal", c_float * 3),
            ("wavelength", c_float * nPhotons),
            ("time", c_float * nPhotons),
            ("contribution", c_float * nPhotons),
        ]

    return HitQueueItem


def createRayQueueItem(nPhotons: int) -> Type[Structure]:
    class RayQueueItem(Structure):
        _fields_ = [
            ("position", c_float * 3),
            ("direction", c_float * 3),
            ("rngStream", c_uint32),
            ("rngCount", c_uint32),
            ("medium", c_uint32 * 2),
            ("wavelength", c_float * nPhotons),
            ("time", c_float * nPhotons),
            ("lin_contrib", c_float * nPhotons),
            ("log_contrib", c_float * nPhotons),
            ("n", c_float * nPhotons),
            ("vg", c_float * nPhotons),
            ("mu_s", c_float * nPhotons),
            ("mu_e", c_float * nPhotons),
        ]

    return RayQueueItem


def createIntersectionQueueItem(nPhotons: int) -> Type[Structure]:
    class IntersectionQueueItem(Structure):
        _fields_ = [
            *createRayQueueItem(nPhotons)._fields_,
            ("geometryIdx", c_int32),
            ("customIdx", c_int32),
            ("triangleIdx", c_int32),
            ("baryc", c_float * 2),
            ("obj2World", c_float * 12),
            ("world2Obj", c_float * 12),
        ]

    return IntersectionQueueItem


def createShadowQueueItem(nPhotons: int) -> Type[Structure]:
    class ShadowQueueItem(Structure):
        _fields_ = [*createRayQueueItem(nPhotons)._fields_, ("dist", c_float)]

    return ShadowQueueItem


def createVolumeScatterQueueItem(nPhotons: int) -> Type[Structure]:
    class VolumeQueueItem(Structure):
        _fields_ = [*createRayQueueItem(nPhotons)._fields_, ("dist", c_float)]

    return VolumeQueueItem
