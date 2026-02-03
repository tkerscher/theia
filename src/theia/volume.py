from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import json

from theia.compiler import createPreamble, loadShader

from importlib.resources.abc import Traversable
from typing import ClassVar, Type


__all__ = [
    "Attenuating",
    "Transparent",
    "VolumeModel",
    "VolumeRNGDraws",
]


def __dir__():
    return __all__


@dataclass
class VolumeRNGDraws:
    """Amount of random numbers needed for each volume model function"""

    sampleInteractionLength: int = 0
    applyVolumeEffect: int = 0
    sampleVolumeInteraction: int = 0
    sampleVolumeScattering: int = 0
    volumeScatterRay: int = 0


class VolumeModel(ABC):
    """
    Base class for all volume models. Volume models implement the interactions
    of a ray as it propagates through volumes containing various media such as
    scattering or absorption.
    """

    models: ClassVar[dict[str, Type[VolumeModel]]] = {}
    """Mapping containing all registered volume models"""
    name: ClassVar[str] = ""
    """Unique extension of this volume model used for serialization"""

    def __init_subclass__(cls: Type[VolumeModel], *, name: str) -> None:
        if not name:
            raise ValueError("extension cannot be empty")
        if name in VolumeModel.models:
            raise RuntimeError(f'"{name}" is already in use by another volume model!')
        VolumeModel.models[name] = cls
        cls.name = name

    def __init__(
        self,
        *,
        rngDraws: VolumeRNGDraws,
        # sbtRecord: bytes | None = None,
        supportsNEE: bool = False,
        supportsParticle: bool = False,
        requiredMediumProperties: set[str] = set(),
    ) -> None:
        self.rngDraws = rngDraws
        # self.sbtRecord = sbtRecord
        self.supportsNEE = supportsNEE
        self.supportsParticle = supportsParticle
        self.requiredMediumProperties = frozenset(requiredMediumProperties)

    @property
    def backwardSourceCode(self) -> str | None:
        """
        Source code implementing volume interaction with backward rays.
        `None` if backward tracing is not supported.
        """
        return None

    @property
    def forwardSourceCode(self) -> str | None:
        """
        Source code implementing volume interaction with forward rays.
        `None` if forward tracing is not supported.
        """
        return None

    def save(self, file) -> None:
        # by default just create a empty file
        file.open("w").close()

    @classmethod
    @abstractmethod
    def load(cls, file: Traversable) -> VolumeModel:
        """Creates a new volume model using the content of the given file"""
        pass

    @staticmethod
    def from_file(file: Traversable) -> VolumeModel:
        if not file.is_file():
            raise RuntimeError(
                f'Cannot not load config from "{file.name}": Not a file!'
            )
        # delegate loading to actual model
        if file.name not in VolumeModel.models:
            raise RuntimeError(f'Unknown volume model "{file.name}"')
        cls = VolumeModel.models[file.name]
        return cls.load(file)

    @abstractmethod
    def __eq__(self, value: object) -> bool:
        pass

    def __hash__(self) -> int:
        # by default use unique name as hash
        return hash(self.name)


class Attenuating(VolumeModel, name="attenuating"):
    """
    Models attenuating and scattering volumes. If `absorb` is `True`, rays get
    randomly absorbed according to the volume's attenuating coefficient.
    Otherwise, rays will be attenuated but never absorbed. The latter does not
    work with particle rays.
    """

    def __init__(self, *, absorb: bool = False) -> None:
        rngDraws = VolumeRNGDraws(
            sampleInteractionLength=1,
            applyVolumeEffect=0,
            sampleVolumeInteraction=2,
            sampleVolumeScattering=2,
            volumeScatterRay=0,
        )
        super().__init__(
            rngDraws=rngDraws,
            supportsNEE=True,
            supportsParticle=True,
            requiredMediumProperties={
                "absorption_coef",
                "scattering_coef",
                "log_phase_function",
                "phase_sampling",
            },
        )
        self._absorb = absorb

    @property
    def isAbsorbing(self) -> bool:
        """Whether rays get absorbed or attenuated"""
        return self._absorb

    @property
    def forwardSourceCode(self) -> str:
        preamble = createPreamble(VOLUME_ABSORB_RAY=self.isAbsorbing)
        code = loadShader("volume/attenuating/forward.glsl")
        return preamble + code

    @property
    def backwardSourceCode(self) -> str:
        preamble = createPreamble(VOLUME_ABSORB_RAY=self.isAbsorbing)
        code = loadShader("volume/attenuating/backward.glsl")
        return preamble + code

    @classmethod
    def load(cls, file: Traversable) -> Attenuating:
        with file.open() as f:
            config = json.load(f)
        return Attenuating(absorb=config["absorb"])

    def save(self, file) -> None:
        config = {"absorb": self.isAbsorbing}
        with file.open("w") as f:
            json.dump(config, f)

    def __eq__(self, value: object) -> bool:
        return type(value) == Attenuating and value.isAbsorbing == self.isAbsorbing

    def __hash__(self) -> int:
        return hash(self.name) ^ hash(self.isAbsorbing)


class Transparent(VolumeModel, name="transparent"):
    """
    Models transparent volumes, i.e. volumes that do not alter the rays that
    pass through it in any way.
    """

    def __init__(self) -> None:
        super().__init__(
            rngDraws=VolumeRNGDraws(),
            supportsNEE=False,
            supportsParticle=True,
        )

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("volume/transparent/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("volume/transparent/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> Transparent:
        return Transparent()

    def __eq__(self, value: object) -> bool:
        return type(value) == Transparent

    def __hash__(self) -> int:
        return hash(self.name)
