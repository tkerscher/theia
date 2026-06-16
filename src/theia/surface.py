from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from theia.compiler import loadShader

from importlib.resources.abc import Traversable
from typing import ClassVar, Type


__all__ = [
    "AbsorbingSurface",
    "BorderSurface",
    "DielectricBeckmannSurface",
    "DielectricMetalSurface",
    "DielectricRoughUnifiedSurface",
    "DielectricSurface",
    "DielectricTrowbridgeReitzShadowedSurface",
    "DielectricTrowbridgeReitzSurface",
    "LambertianReflectingSurface",
    "SurfaceModel",
    "SurfaceRNGDraws",
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
    

class DielectricMetalSurface(SurfaceModel, name="dielectric_metal"):
    """
    Models transmission and reflection of perfectly specular surfaces between
    a dielectric medium and a metal. 
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
            requiredMaterialProperties={"reflectivity"},
        )

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("surface/dielectric_metal/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("surface/dielectric_metal/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> DielectricMetalSurface:
        return DielectricMetalSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == DielectricMetalSurface

    def __hash__(self) -> int:
        return hash(self.name)


class DielectricRoughUnifiedSurface(SurfaceModel, name="dielectric_unified"):
    """
    Models transmission and reflection of rough surfaces between two dielectric
    media. This model mirrors the Geant4 UNIFIED model with ground finish. This
    expexts the surface material to have the percent point function of the 
    microfacet distribution as a table property, which can be created using
    `property.RoughUnifiedSurfacePPFProperty`.

    """

    def __init__(self) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=34,
            sampleSurfaceInteraction=2,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"refractive_index"},
            requiredMaterialProperties={"ppf", "prob_backscatter", "prob_specularspike", "prob_specularlobe", "prob_diffuselobe"},
        )

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("surface/dielectric_rough_unified/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("surface/dielectric_rough_unified/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> DielectricRoughUnifiedSurface:
        return DielectricRoughUnifiedSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == DielectricRoughUnifiedSurface

    def __hash__(self) -> int:
        return hash(self.name)



class DielectricBeckmannSurface(SurfaceModel, name="dielectric_beckmann"):
    """
    Models transmission and reflection of rough surfaces between two dielectric
    media. The surface is modelled as consisting of micro-facets. Micro-facet normals 
    are sampled according to the Beckmann model. No shadowing or masking of micro-facets 
    is applied.

    Note
    ----
    The Geant4 UNIFIED SpecularLobe is essentially the small-angle approximation of 
    the Beckmann model with `sigma_alpha = roughness_parameter / sqrt(2)`. For small
    roughnesses, these models will give approximately the same result.
    """

    def __init__(self) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=25,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"refractive_index"},
            requiredMaterialProperties={"roughness_parameter"},
        )

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("surface/dielectric_rough/beckmann/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("surface/dielectric_rough/beckmann/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> DielectricBeckmannSurface:
        return DielectricBeckmannSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == DielectricBeckmannSurface

    def __hash__(self) -> int:
        return hash(self.name)


class DielectricTrowbridgeReitzSurface(SurfaceModel, name="dielectric_trowbridge_reitz"):
    """
    Models transmission and reflection of rough surfaces between two dielectric
    media. The surface is modelled as consisting of micro-facets. Micro-facet normals 
    are sampled according to the Trowbridge-Reitz (GGX) model. No shadowing or masking 
    of micro-facets is applied.
    """

    def __init__(self) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=25,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"refractive_index"},
            requiredMaterialProperties={"roughness_parameter"},
        )

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("surface/dielectric_rough/trowbridge_reitz/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("surface/dielectric_rough/trowbridge_reitz/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> DielectricTrowbridgeReitzSurface:
        return DielectricTrowbridgeReitzSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == DielectricTrowbridgeReitzSurface

    def __hash__(self) -> int:
        return hash(self.name)


class DielectricTrowbridgeReitzShadowedSurface(SurfaceModel, name="dielectric_trowbridge_reitz_shadowed"):
    """
    Models transmission and reflection of rough surfaces between two dielectric
    media. The surface is modelled as consisting of micro-facets. Micro-facet normals 
    are sampled from the visible distribution of the Trowbridge-Reitz (GGX) model, with 
    Smith masking applied to the outgoing ray.
    """

    def __init__(self) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=33,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"refractive_index"},
            requiredMaterialProperties={"roughness_parameter"},
        )

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("surface/dielectric_rough/trowbridge_reitz_shadowed/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("surface/dielectric_rough/trowbridge_reitz_shadowed/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> DielectricTrowbridgeReitzShadowedSurface:
        return DielectricTrowbridgeReitzShadowedSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == DielectricTrowbridgeReitzShadowedSurface

    def __hash__(self) -> int:
        return hash(self.name)


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
