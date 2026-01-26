from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from importlib.resources.abc import Traversable
from typing import ClassVar, Type


@dataclass
class SurfaceRNGDraws:
    """Amount of random numbers drawn for each surface model function"""

    processSurfaceInteraction: int
    processSurfaceTarget: int


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
        rngDraws: SurfaceRNGDraws,
        # sbtRecord: bytes | None = None,
        # supportsNEE: bool = False,
        requiredMediumProperties: set[str] = set(),
        requiredMaterialProperties: set[str] = set(),
    ) -> None:
        self.rngDraws = rngDraws
        # self.sbtRecord = sbtRecord
        # self.supportsNEE = supportsNEE
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


class DielectricSurface(SurfaceModel, name="dielectric"):
    """
    Models transmission and reflection of perfectly specular surfaces between
    two dielectric media.
    """

    def __init__(self) -> None:
        draws = SurfaceRNGDraws(
            processSurfaceInteraction=0,
            processSurfaceTarget=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"refractive_index"},
        )

    @property
    def backwardSourceCode(self) -> str:
        raise NotImplementedError

    @property
    def forwardSourceCode(self) -> str:
        raise NotImplementedError

    @classmethod
    def load(cls, file: Traversable) -> DielectricSurface:
        return DielectricSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == DielectricSurface

    def __hash__(self) -> int:
        return hash(self.name)
