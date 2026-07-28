from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import json

from theia.compiler import createPreamble, loadShader

from importlib.resources.abc import Traversable
from typing import ClassVar, Type


__all__ = [
    "AbsorbingSurface",
    "BorderSurface",
    "DielectricSurface",
    "LambertianReflectingSurface",
    "MetallicSurface",
    "SurfaceModel",
    "SurfaceRNGDraws",
    "ThinDielectricSurface",
    "ThinMetallicSurface",
]


def __dir__():
    return __all__


@dataclass
class SurfaceRNGDraws:
    """Amount of random numbers drawn for each surface model function"""

    prepareSurface: int
    sampleSurfaceInteraction: int
    processSurfaceTargetHit: int

    # MIS

    surfaceScatterRay: int = 0
    sampleSurfaceScattering: int = 0


class SurfaceModel(ABC):
    """
    Base class for all surface models. Surface models implement the interactions
    of a ray with a surface as it hits a geometry.
    """

    models: ClassVar[dict[str, Type[SurfaceModel]]] = {}
    """Mapping containing all registered surface models"""
    name: ClassVar[str] = ""

    def __init_subclass__(cls, *, name: str) -> None:
        if not name:
            raise ValueError("Surface model name cannot be empty")
        if name in SurfaceModel.models:
            raise RuntimeError(f'"{name}" is already in use by another surface model')
        SurfaceModel.models[name] = cls
        cls.name = name

    def __init__(
        self,
        *,
        rngDraws: SurfaceRNGDraws | None,
        # sbtRecord: bytes | None = None,
        supportsMIS: bool = False,
        requiredMediumProperties: set[str] = set(),
        requiredMaterialProperties: set[str] = set(),
    ) -> None:
        if rngDraws is None:
            rngDraws = SurfaceRNGDraws(
                prepareSurface=0,
                sampleSurfaceInteraction=0,
                processSurfaceTargetHit=0,
            )
        self.rngDraws = rngDraws
        # self.sbtRecord = sbtRecord
        self.supportsMIS = supportsMIS
        self.requiredMediumProperties = frozenset(requiredMediumProperties)
        self.requiredMaterialProperties = frozenset(requiredMaterialProperties)

    @property
    def backwardSourceCode(self) -> str | None:
        """
        Source code implementing surface interaction with backward rays.
        `None` if backward tracing is not supported.
        """
        return None

    @property
    def forwardSourceCode(self) -> str | None:
        """
        Source code implementing surface interaction with forward rays.
        `None` if forward tracing is not supported.
        """
        return None

    def save(self, file) -> None:
        # by default just create an empty file
        file.open("w").close()

    @classmethod
    @abstractmethod
    def load(cls, file: Traversable) -> SurfaceModel:
        """Creates a new surface model using the content of the given file"""
        pass

    @staticmethod
    def from_file(file: Traversable) -> SurfaceModel:
        if not file.is_file():
            raise RuntimeError(f'Cannot load config from "{file.name}": Not a file!')
        if file.name not in SurfaceModel.models:
            raise RuntimeError(f'Unknown surface model "{file.name}"')
        # delegate loading to actual model
        cls = SurfaceModel.models[file.name]
        return cls.load(file)

    @abstractmethod
    def __eq__(self, value: object) -> bool:
        pass

    def __hash__(self) -> int:
        return hash(self.name)


class AbsorbingSurface(SurfaceModel, name="absorbing"):
    """Models perfectly absorbing surfaces and detectors."""

    def __init__(self) -> None:
        super().__init__(rngDraws=None)

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("surface/absorbing/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("surface/absorbing/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> AbsorbingSurface:
        return AbsorbingSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == AbsorbingSurface

    def __hash__(self) -> int:
        return hash(self.name)


class BorderSurface(SurfaceModel, name="border"):
    """Models border between two surfaces. Transmits all rays without altering them."""

    def __init__(self) -> None:
        super().__init__(rngDraws=None)

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("surface/border/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("surface/border/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> BorderSurface:
        return BorderSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == BorderSurface

    def __hash__(self) -> int:
        return hash(self.name)


class DielectricSurface(SurfaceModel, name="dielectric"):
    """
    Models transmission and reflection of perfectly specular surfaces between
    two dielectric media.
    """

    def __init__(self) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=1,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"refractive_index"},
        )

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("surface/dielectric/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("surface/dielectric/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> DielectricSurface:
        return DielectricSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == DielectricSurface

    def __hash__(self) -> int:
        return hash(self.name)


class MetallicSurface(SurfaceModel, name="metallic"):
    """
    Models absorption and specular reflection on a metallic surface.
    Reflectivity can either be provided directly or is calculated using the
    complex refractive index of the metal.

    If `absorb` is set to `True`, rays may be absorbed to stop tracing.
    Otherwise they will always reflect attenuated by the surface's reflectance
    if it has non-zero reflectance.
    """

    def __init__(self, *, absorb: bool = False) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=1,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMaterialProperties={"reflectivity"},
            requiredMediumProperties={"refractive_index", "imag_refractive_index"},
        )
        self._absorb = absorb

    @property
    def isAbsorbing(self) -> bool:
        """Whether ray may be absorbed to stop tracing"""
        return self._absorb

    @property
    def forwardSourceCode(self) -> str:
        preamble = createPreamble(SURFACE_ABSORB_RAY=self.isAbsorbing)
        return loadShader("surface/metallic/forward.glsl", preamble)

    @property
    def backwardSourceCode(self) -> str:
        preamble = createPreamble(SURFACE_ABSORB_RAY=self.isAbsorbing)
        return loadShader("surface/metallic/backward.glsl", preamble)

    @classmethod
    def load(cls, file: Traversable) -> MetallicSurface:
        with file.open() as f:
            config = json.load(f)
        return MetallicSurface(absorb=config["absorb"])

    def save(self, file) -> None:
        config = {"absorb": self.isAbsorbing}
        with file.open("w") as f:
            json.dump(config, f)

    def __eq__(self, value: object) -> bool:
        return type(value) == MetallicSurface and value.isAbsorbing == self.isAbsorbing

    def __hash__(self) -> int:
        return hash(self.name) ^ hash(self.isAbsorbing)


class LambertianReflectingSurface(SurfaceModel, name="lambert"):
    """
    Models perfectly diffuse reflecting surfaces.
    Reflectance defaults to 100%, but can be specified as function of wavelength
    in the `REFLECTIVITY` property.
    """

    def __init__(self) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=1,
            sampleSurfaceInteraction=2,
            processSurfaceTargetHit=0,
            sampleSurfaceScattering=2,
        )
        super().__init__(
            rngDraws=draws,
            requiredMaterialProperties={"reflectivity"},
            supportsMIS=True,
        )

    @property
    def backwardSourceCode(self) -> str | None:
        return loadShader("surface/lambert/backward.glsl")

    @property
    def forwardSourceCode(self) -> str | None:
        return loadShader("surface/lambert/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> LambertianReflectingSurface:
        return LambertianReflectingSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == LambertianReflectingSurface

    def __hash__(self) -> int:
        return hash(self.name)


class ThinDielectricSurface(SurfaceModel, name="thin_dielectric"):
    """
    Models transmission and reflection through a thin dielectric layer between
    two other dielectric media. Optionally includes interference effects by
    setting `interference` to `True`.
    """

    def __init__(self, *, interference=False) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=1,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"refractive_index"},
            requiredMaterialProperties={"layer_thickness"},
        )
        self._interference = interference

    @property
    def isInterfering(self) -> bool:
        """Whether interference effects are included"""
        return self._interference

    @property
    def forwardSourceCode(self) -> str:
        preamble = createPreamble(
            SURFACE_THIN_LAYER_INCLUDE_INTERFERENCE=self.isInterfering
        )
        shader = loadShader("surface/thin/dielectric/forward.glsl")
        return preamble + shader

    @property
    def backwardSourceCode(self) -> str:
        preamble = createPreamble(
            SURFACE_THIN_LAYER_INCLUDE_INTERFERENCE=self.isInterfering
        )
        shader = loadShader("surface/thin/dielectric/backward.glsl")
        return preamble + shader

    @classmethod
    def load(cls, file: Traversable) -> ThinDielectricSurface:
        with file.open() as f:
            config = json.load(f)
        return ThinDielectricSurface(interference=config["interference"])

    def save(self, file) -> None:
        config = {"interference": self.isInterfering}
        with file.open("w") as f:
            json.dump(config, f)

    def __eq__(self, value: object) -> bool:
        return (
            type(value) == ThinDielectricSurface
            and value.isInterfering == self.isInterfering
        )

    def __hash__(self) -> int:
        return hash(self.name) ^ hash(self.isInterfering)


class ThinMetallicSurface(SurfaceModel, name="thin_metallic"):
    """
    Models transmission, reflection and absorption in a thin metallic layer
    between two dielectric media including interference effects within the thin
    layer.

    When used as detector surface, will report absorbed photons as detected.
    If `absorb` is set to `True`, rays may be absorbed to stop tracing.
    """

    def __init__(self, *, absorb=True) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=1,
            sampleSurfaceInteraction=1,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"imag_refractive_index", "refractive_index"},
            requiredMaterialProperties={"layer_medium_idx", "layer_thickness"},
        )
        self._absorb = absorb

    @property
    def isAbsorbing(self) -> bool:
        """Whether rays may be absorbed in the thin layer"""
        return self._absorb

    @property
    def forwardSourceCode(self) -> str:
        preamble = createPreamble(SURFACE_ABSORB_RAY=self.isAbsorbing)
        shader = loadShader("surface/thin/metallic/forward.glsl")
        return preamble + shader

    @property
    def backwardSourceCode(self) -> str:
        preamble = createPreamble(SURFACE_ABSORB_RAY=self.isAbsorbing)
        shader = loadShader("surface/thin/metallic/backward.glsl")
        return preamble + shader

    @classmethod
    def load(cls, file: Traversable) -> ThinMetallicSurface:
        with file.open() as f:
            config = json.load(f)
        return ThinMetallicSurface(absorb=config["absorb"])

    def save(self, file) -> None:
        config = {"absorb": self.isAbsorbing}
        with file.open("w") as f:
            json.dump(config, f)

    def __eq__(self, value: object) -> bool:
        return (
            type(value) == ThinMetallicSurface and value.isAbsorbing == self.isAbsorbing
        )

    def __hash__(self) -> int:
        return hash(self.name) ^ hash(self.isAbsorbing)
