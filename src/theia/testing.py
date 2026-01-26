from __future__ import annotations

import hephaistos as hp

from hephaistos.pipeline import PipelineStage
from hephaistos.queue import IOQueue

from ctypes import Structure, c_float, c_uint32
from hephaistos.glsl import buffer_reference, vec3
from theia.util import createCType

from theia.compiler import createPreamble, compileShader
from theia.light import LightSource
from theia.material import MaterialStore, VACUUM_IDX
from theia.random import RNG
from theia.ray import RayModel
import theia.units as u


class BackwardLightSampler(PipelineStage):

    class SamplerParams(Structure):
        _fields_ = [
            ("_queueAdr", buffer_reference),
            ("_count", c_uint32),
            ("_mediumIdx", c_uint32),
            ("observer", vec3),
        ]

    def __init__(
        self,
        batchSize: int,
        source: LightSource,
        ray: RayModel,
        *,
        rng: RNG,
        materials: MaterialStore | None = None,
        medium: str | int | None = None,
        observer: tuple[float, float, float] | None = None,
        box_size: float = 100.0 * u.m,
    ) -> None:
        super().__init__(
            {"SamplerParams": BackwardLightSampler.SamplerParams},
            {"medium"},
        )
        # check params
        if not source.supportBackward:
            raise ValueError("Light source does not support backward mode")
        # process params
        if isinstance(medium, str):
            if materials is None:
                raise ValueError("Medium cannot be locked up if no store is provided")
            medium = materials.media[medium]
        elif medium is None:
            medium = VACUUM_IDX
        if materials is None:
            materials = MaterialStore([], mediaSlots=ray.requiredMediumProperties)
        if observer is None:
            observer = (float("NaN"), float("NaN"), float("NaN"))

        # allocate queue
        resultFields = [
            ("observer", c_float * 3),
            ("normal", c_float * 3),
            ("wavelength", c_float),
        ]
        resultFields.extend(ray.forwardItem._fields_)
        item = createCType("Item", resultFields)
        self._queue = IOQueue(item, batchSize, mode="retrieve", skipCounter=True)
        assert self._queue.tensor is not None
        # save params
        self._ray = ray
        self._rng = rng
        self._source = source
        self._materials = materials
        self.setParams(
            _queueAdr=self._queue.tensor.address,
            _count=batchSize,
            _mediumIdx=medium,
            observer=observer,
        )

        # create program
        lamSource = source.wavelengthSource
        preamble = createPreamble(DIM=0.5 * box_size)
        headers = {
            "ray.glsl": ray.sourceCode,
            "rng.glsl": rng.sourceCode,
            "source.glsl": source.sourceCode,
            "photon.glsl": "" if lamSource is None else lamSource.sourceCode,
            **materials.header,
        }
        code = compileShader("lightsource/sample/backward.glsl", preamble, headers)
        self._program = hp.Program(code)
        # bind memory
        materials.bindParams(self._program)

    @property
    def batchSize(self) -> int:
        """Number of samples to draw per batch"""
        return self.getParam("_count")

    @property
    def materials(self) -> MaterialStore:
        """Optional material store in use"""
        return self._materials

    @property
    def medium(self) -> int:
        """Index of the medium the light source is submerged in"""
        return self.getParam("_medium")

    @medium.setter
    def medium(self, value: str | int | None) -> None:
        if self._materials is None and value is not None:
            raise ValueError("Cannot set material if none were provide!")
        if value is None:
            value = VACUUM_IDX
        elif isinstance(value, str):
            assert self._materials is not None  # to make the linter happy...
            value = self._materials.media[value]
        self.setParam("_mediumIdx", value)

    @property
    def queue(self) -> IOQueue:
        """Queue holding the sampled light rays"""
        return self._queue

    @property
    def rayModel(self) -> RayModel:
        """Ray model used for sampling"""
        return self._ray

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    @property
    def source(self) -> LightSource:
        """Source generating light rays"""
        return self._source

    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all stages involved with this sampler in the correct
        order suitable for creating a pipeline.
        """
        stages: list[PipelineStage] = [self.rng]
        if self.source.wavelengthSource is not None:
            stages.append(self.source.wavelengthSource)
        stages.extend([self.source, self])
        return stages

    def run(self, i: int) -> list[hp.Command]:
        for stage in self.collectStages():
            stage.bindParams(self._program, i)
        groups = -(self.batchSize // -512)
        return [
            self._program.dispatch(groups),
            *self.queue.run(i),
        ]
