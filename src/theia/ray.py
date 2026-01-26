from __future__ import annotations

from abc import ABC, abstractmethod
from ctypes import Structure, c_float, c_int32, c_uint32
from theia.compiler import createPreamble, loadShader
from theia.util import createCType

__all__ = [
    "RayModel",
    "UnpolarizedRay",
]


def __dir__():
    return __all__


class RayModel(ABC):
    """
    Base class for ray models to be used by tracers. They hold the current state
    of rays as they are propagated by the tracer as well as all the necessary
    information to produce a hit.

    Parameters
    ----------
    transient: bool
        If `True`, the ray also simulates the amount of time needed for the ray
        to propagate.
    particle: bool
        If `True`, a single ray equals a single particle. It therefore cannot
        model a continuous amount of expected particles, which may have
        implications for the tracer.
    supportsForward: bool
        Whether the ray model supports propagation in forward direction
    supportsBackward: bool
        Whether the ray model supports propagation in backward direction
    requiredMediumProperties: set[str], default={}
        Name of medium properties the ray model relies on and thus must be
        present
    """

    def __init__(
        self,
        *,
        transient: bool,
        particle: bool,
        supportsForward: bool,
        supportsBackward: bool,
        requiredMediumProperties: set[str] = set(),
    ) -> None:
        self.isTransient = transient
        self.isParticle = particle
        self.supportsForward = supportsForward
        self.supportsBackward = supportsBackward
        self.requiredMediumProperties = frozenset(requiredMediumProperties)

    @property
    def backwardItem(self) -> type[Structure] | None:
        """
        Optional structure describing the fields and memory layout of backward
        rays. May be `None`, if backward tracing is not supported.
        """
        return None

    @property
    def cameraItem(self) -> type[Structure] | None:
        """
        Optional structure describing the fields and memory layout of camera
        items consisting of a backward ray followed by a camera hit. May be
        `None` if backward tracing is not supported.
        """
        return None

    @property
    @abstractmethod
    def forwardItem(self) -> type[Structure]:
        """Describes the fields and memory layout of forward rays."""
        pass

    @property
    @abstractmethod
    def hitItem(self) -> type[Structure]:
        """Describes the fields and memory layout of a hit item."""
        pass

    @property
    @abstractmethod
    def sourceCode(self) -> str:
        """Source code of the ray model."""
        pass


class UnpolarizedRay(RayModel):
    """
    Simple unpolarized ray. If `particle` is `True`, a single ray corresponds to
    a single photon. Otherwise, a ray may hold a rational number of expected
    photons, allowing tracers to use more sophisticated algorithms.
    """

    def __init__(self, *, transient: bool = True, particle: bool = False) -> None:
        super().__init__(
            transient=transient,
            particle=particle,
            supportsForward=True,
            supportsBackward=not particle,
            requiredMediumProperties={"group_velocity"} if transient else set(),
        )

        # assemble items
        hitFields = [
            ("position", c_float * 3),
            ("direction", c_float * 3),
            ("normal", c_float * 3),
            ("objectId", c_int32),
            ("wavelength", c_float),
        ]
        forwardFields = [
            ("position", c_float * 3),
            ("direction", c_float * 3),
            ("wavelength", c_float),
            ("mediumIdx", c_uint32),
        ]
        backwardFields = [
            ("position", c_float * 3),
            ("direction", c_float * 3),
            ("wavelength", c_float),
            ("mediumIdx", c_uint32),
        ]
        cameraHitFields = [
            ("hitPosition", c_float * 3),
            ("hitDirection", c_float * 3),
            ("hitNormal", c_float * 3),
            ("objectId", c_int32),
        ]
        if transient:
            hitFields.append(("time", c_float))
            forwardFields.append(("time", c_float))
            backwardFields.append(("time", c_float))
        if not particle:
            hitFields.append(("contrib", c_float))
            forwardFields.append(("contrib", c_float))
            backwardFields.append(("contrib", c_float))

        self._hitItem = createCType("HitItem", hitFields)
        self._forwardItem = createCType("ForwardItem", forwardFields)
        if particle:
            self._backwardItem = None
            self._cameraItem = None
        else:
            self._backwardItem = createCType("BackwardItem", backwardFields)
            cameraFields = backwardFields + cameraHitFields
            self._cameraItem = createCType("CameraItem", cameraFields)

    @property
    def backwardItem(self) -> type[Structure] | None:
        return self._backwardItem

    @property
    def cameraItem(self) -> type[Structure] | None:
        return self._cameraItem

    @property
    def forwardItem(self) -> type[Structure]:
        return self._forwardItem

    @property
    def hitItem(self) -> type[Structure]:
        return self._hitItem

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            RAY_TRANSIENT=self.isTransient,
            RAY_PARTICLE=self.isParticle,
        )
        code = loadShader("ray/unpolarized.glsl")
        return preamble + code
