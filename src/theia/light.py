from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats.sampling import NumericalInversePolynomial

import hephaistos as hp
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import IOQueue

from ctypes import Structure, c_float, c_int64, c_uint32
from hephaistos.glsl import buffer_reference, vec2, vec3
from theia.util import createCType

from theia.compiler import compileShader, createPreamble, loadShader
from theia.lookup import uploadTables
from theia.material import MaterialStore
from theia.random import RNG
from theia.ray import RayModel
import theia.units as u

from warnings import warn

from collections.abc import Callable
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ConeLightSource",
    "ConstWavelengthSource",
    "FunctionWavelengthSource",
    "HostLightSource",
    "HostWavelengthSource",
    "LightSampler",
    "LightSource",
    "PencilLightSource",
    "SphericalLightSource",
    "StreamingHostLightSource",
    "StreamingHostWavelengthSource",
    "UniformWavelengthSource",
    "WavelengthSource",
]


def __dir__():
    return __all__


class WavelengthSource(SourceCodeMixin):
    """
    Base class for samplers producing wavelengths.
    """

    name = "Wavelength Sampler"

    def __init__(
        self,
        *,
        nRNGSamples: int = 0,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self._nRNGSamples = nRNGSamples

    @property
    def nRNGSamples(self) -> int:
        """Amount of random numbers drawn per sample"""
        return self._nRNGSamples


class HostWavelengthSource(WavelengthSource):
    """
    Wavelength source passing samples from CPU to GPU.

    Parameters
    ----------
    capacity: int
        Maximum number of samples that can be drawn per run
    emitParticle: int
        If `False`, samples have a dedicated contribution value.
    updateFn: (HostWavelengthSource, int) -> None | None, default=None
        Optional update function called before the pipeline processes a task.
        `i` is the i-th configuration the update should affect.
        Can be used to stream in new samples on demand.
    extra: set[str], default={}
        Set of extra parameter name, that can be set and retrieved using the
        stage api.
    """

    class WavelengthParams(Structure):
        _fields_ = [("_queueAdr", buffer_reference), ("_queueSize", c_uint32)]

    def __init__(
        self,
        capacity: int,
        *,
        emitParticle: bool = False,
        updateFn: Callable[[HostWavelengthSource, int], None] | None = None,
        extra: set[str] = set(),
    ) -> None:
        super().__init__(
            params={"WavelengthParams": HostWavelengthSource.WavelengthParams},
            extra=extra,
        )
        # allocate memory
        fields = [("wavelength", c_float)]
        if not emitParticle:
            fields.append(("contrib", c_float))
        item = createCType("Item", fields)
        self._queue = IOQueue(item, capacity, mode="update", skipCounter=True)
        assert self._queue.tensor is not None
        # save params
        self._capacity = capacity
        self._emitParticle = emitParticle
        self._updateFn = updateFn
        self.setParams(_queueAdr=self._queue.tensor.address, _queueSize=capacity)

    @property
    def capacity(self) -> int:
        """Maximum number of samples that can be drawn per run"""
        return self._capacity

    @property
    def emitsParticle(self) -> bool:
        """If `False`, samples have a dedicated contribution value."""
        return self._emitParticle

    @property
    def queue(self) -> IOQueue:
        """Queue containing the data for the next batch"""
        return self._queue

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(WAVELENGTH_SOURCE_EMIT_PARTICLE=self.emitsParticle)
        code = loadShader("wavelengthsource/host.glsl")
        return preamble + code

    def update(self, i: int) -> None:
        if self._updateFn is not None:
            self._updateFn(self, i)
        super().update(i)

    def run(self, i: int) -> list[hp.Command]:
        return [*self.queue.run(i), *super().run(i)]


class StreamingHostWavelengthSource(HostWavelengthSource):
    """
    Wavelength source using a provided data source to sequentially feed pre
    sampled wavelengths to the device. Subsequent batches move forward into the
    provided data streaming it to the device. Providing new data resets the
    stream.

    Parameters
    ----------
    batchSize: int
        Number of wavelengths to sample per batch
    data: dict[str, ArrayLike] | None, default=None
        Data source from which the batches are produced
    emitParticle: bool, default=False
        Whether the source should produce particles or not.

    Stage Parameters
    ----------------
    data: ArrayLike | None
        Data source from which the batches are produced.

    Note
    ----
    There is no special handling for when the last batch from the data is not
    filling the queue completely. Data in the queue beyond the end is likely old
    or garbage, but the source does not prevent the tracer from reading it.
    """

    def __init__(
        self,
        batchSize: int,
        *,
        data: dict[str, ArrayLike] | None = None,
        emitParticle: bool = False,
    ) -> None:
        super().__init__(
            batchSize,
            emitParticle=emitParticle,
            updateFn=self._updateData,
            extra={"data"},
        )
        self._dataFields = set(self.queue.view(0).fields)
        self.data = data

    @property
    def data(self) -> dict[str, NDArray] | None:
        """Data used as streaming source"""
        return self._data

    @data.setter
    def data(self, value: dict[str, ArrayLike] | None) -> None:
        self._offset = 0
        self._dataSize = 0
        if value is None:
            self._data = None
            return
        # check if all the needed fields are there
        if not self._dataFields.issubset(value.keys()):
            m = self._dataFields.difference(value.keys())
            raise ValueError(f"provided data misses expected fields: {m}")
        self._dataSize = np.size(value["wavelength"], 0)
        # ensure all fields have the same size
        for field in self._dataFields:
            if np.size(value[field], 0) != self._dataSize:
                raise ValueError("fields of provided data differ in size!")
            # TODO: Check correct dimensions, too?
        # create ndarray and store them for later
        self._data = {f: np.asarray(value[f]) for f in self._dataFields}

    @data.deleter
    def data(self) -> None:
        self.data = None

    @staticmethod
    def _updateData(source: HostWavelengthSource, i: int) -> None:
        assert isinstance(source, StreamingHostWavelengthSource)
        if source.data is None:
            warn(
                "StreamingHostWavelengthSource: "
                "New data was requested but no source was given!"
            )
            return
        if source._offset >= source._dataSize:
            warn("StreamingHostWavelengthSource ran out of data!")
            return

        # calculate next data range
        start = source._offset
        end = min(source._offset + source.capacity, source._dataSize)
        count = end - start
        source._offset = end
        # copy new data block
        view = source.queue.view(i)
        for field in source._dataFields:
            view[field][:count] = source.data[field][start:end]


class ConstWavelengthSource(WavelengthSource):
    """
    Sampler generating photons of constant wavelength.

    Parameters
    ----------
    wavelength: float, default=600.0nm
        Constant wavelength

    Stage Parameters
    ----------------
    wavelength: float, default=600.0nm
        Constant wavelength
    """

    name = "Const Wavelength Source"

    class WavelengthParams(Structure):
        _fields_ = [("wavelength", c_float)]

    def __init__(self, wavelength: float = 600.0 * u.nm) -> None:
        super().__init__(params={"WavelengthParams": self.WavelengthParams})
        self.setParams(wavelength=wavelength)

    @property
    def sourceCode(self) -> str:
        return loadShader("wavelengthsource/const.glsl")


class UniformWavelengthSource(WavelengthSource):
    """
    Sampler generating photons uniform in wavelength and time.

    Parameters
    ----------
    lambdaRange: (float, float), default=(300.0, 700.0)
        min and max wavelength the source emits
    normalize: bool, default=True
        If True, adds an extra factor of one over the lambdaRange to the
        contribution to normalize the light source.

    Stage Parameters
    ----------------
    lambdaRange: (float, float), default=(300.0, 700.0)
        min and max wavelength the source emits
    normalize: bool, default=True
        If True, sets the contribution per sampled wavelength to 1.0. Normally,
        this would be the inverse sample probability, which here equals to the
        lambda range.

    Note
    ----
    In combination with light sources that allow settings the total amount of
    photons per wavelength via e.g. a parameter `budget`, setting `normalize` to
    `True` will make `budget` the total amount of photons regardless of
    `lambdaRange`.
    """

    class SourceParams(Structure):
        _fields_ = [
            ("lambdaRange", vec2),
            ("_contrib", c_float),
        ]

    def __init__(
        self,
        *,
        lambdaRange: tuple[float, float] = (300.0, 700.0),
        normalize: bool = True,
    ) -> None:
        super().__init__(
            nRNGSamples=1,
            params={"WavelengthParams": self.SourceParams},
            extra={"normalize"},
        )
        # save params
        self.setParams(lambdaRange=lambdaRange, normalize=normalize)

    @property
    def normalize(self) -> bool:
        """Whether to normalize this source."""
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool) -> None:
        self._normalize = value

    @property
    def sourceCode(self) -> str:
        return loadShader("wavelengthsource/uniform.glsl")

    def _finishParams(self, i: int) -> None:
        c = 1.0
        lr = self.getParam("lambdaRange")
        lr = lr[1] - lr[0]
        if lr != 0.0 and not self.normalize:
            c *= abs(lr)
        self.setParam("_contrib", c)


class FunctionWavelengthSource(WavelengthSource):
    """
    Wavelength source generating samples by importance sampling the given
    function or distribution. Numerically integrates and inverts the
    distribution before discretizing the result into a look up table used by
    the GPU.

    Parameters
    ----------
    fn: (float) -> float
        Function to be importance sampled, mapping wavelengths in nm to a
        scalar.
    lambdaRange: (float, float), default=(300.0, 700.0)nm
        Wavelength range of the generated samples.
    numSamples: int, default=1024
        Number of entries in the final look up table.
    """

    name = "Function Wavelength Source"

    class WavelengthParams(Structure):
        _fields_ = [("_table", c_int64), ("_contrib", c_float)]

    def __init__(
        self,
        fn: Callable[[float], float],
        *,
        lambdaRange: tuple[float, float] = (300.0, 700.0) * u.nm,
        numSamples: int = 1024,
    ) -> None:
        super().__init__(
            nRNGSamples=1,
            params={"WavelengthParams": self.WavelengthParams},
        )
        self._updateFn(fn, lambdaRange, numSamples)

    @property
    def sourceCode(self) -> str:
        return loadShader("wavelengthsource/function.glsl")

    def _updateFn(
        self,
        fn: Callable[[float], float],
        lambdaRange: tuple[float, float],
        numSamples: int,
    ) -> None:
        # integrate fn to get constant contribution
        contrib, _ = quad(fn, *lambdaRange)

        class Dist:
            """Bundle fn into a class for use with scipy"""

            def pdf(self, x):
                return fn(x)

        # invert cdf
        inv_cdf = NumericalInversePolynomial(Dist(), domain=lambdaRange)
        # sample inverted cdf
        u = np.linspace(0.0, 1.0, numSamples)
        x = inv_cdf.ppf(u)
        # create table from samples
        self._tableMemory, [table_adr] = uploadTables([x])

        # update params
        self.setParams(_table=table_adr, _contrib=contrib)


class LightSource(PipelineStage):
    """Base class for samplers producing light rays"""

    name = "Light Source"

    def __init__(
        self,
        nRNGForward: int = 0,
        nRNGBackward: int = 0,
        wavelengthSource: WavelengthSource | None = None,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self._rngForward = nRNGForward
        self._rngBackward = nRNGBackward
        self._wavelengthSource = wavelengthSource

    @property
    def backwardSourceCode(self) -> str | None:
        """
        Source code implementing light source sampling in backward tracing.
        `None` if backward tracing is not supported.
        """
        return None

    @property
    def forwardSourceCode(self) -> str | None:
        """
        Source code implementing light source sampling in forward tracing.
        `None` if forward tracing is not supported.
        """
        return None

    @property
    def nRNGForward(self) -> int:
        """Amount of random numbers drawn per sample in forward mode"""
        return self._rngForward

    @property
    def nRNGBackward(self) -> int:
        """Amount of random numbers drawn per sample in backward mode"""
        return self._rngBackward

    @property
    def wavelengthSource(self) -> WavelengthSource | None:
        """Optional wavelength source used for sampling wavelengths"""
        return self._wavelengthSource

    def run(self, i: int) -> list[hp.Command]:
        # most light sources will not issue any commands
        # so an empty list is a sensible default
        return []


class LightSampler(PipelineStage):
    """
    Samples a given light source and puts the result into a queue for processing
    on the CPU.

    Parameters
    ----------
    batchSize: int
        Number of samples to draw per batch.
    source: LightSource
        Light source to sample from.
    ray: RayModel
        Model describing the structure of rays to sample.
    rng: RNG | None, default=None
        Optional RNG the light source may require.
    materials: MaterialStore | None, default=None
        Optional material store the light source or ray model may require.
    """

    name = "Light Sampler"

    class SampleParams(Structure):
        _fields_ = [
            ("_batchSize", c_uint32),
            ("baseCount", c_uint32),
            ("_queueAdr", buffer_reference),
            ("_queueSize", c_uint32),
        ]

    def __init__(
        self,
        batchSize: int,
        source: LightSource,
        ray: RayModel,
        *,
        rng: RNG | None = None,
        materials: MaterialStore | None = None,
    ) -> None:
        # check if we have RNG if needed
        if source.forwardSourceCode is None:
            raise ValueError("Light source does not support forward mode")
        if source.nRNGForward > 0 and rng is None:
            raise ValueError("Light source requires a RNG but none was given")
        super().__init__({"SampleParams": LightSampler.SampleParams})
        # create queue
        self._queue = IOQueue(
            ray.forwardItem, batchSize, mode="retrieve", skipCounter=True
        )
        assert self._queue.tensor is not None
        # create empty material store if needed
        if materials is None:
            materials = MaterialStore([], mediaSlots=ray.requiredMediumProperties)
        # save params
        self._source = source
        self._ray = ray
        self._rng = rng
        self._materials = materials
        self.setParams(
            _batchSize=batchSize,
            baseCount=0,
            _queueAdr=self._queue.tensor.address,
            _queueSize=batchSize,
        )

        # create code
        lamSource = source.wavelengthSource
        headers = {
            "light.glsl": source.forwardSourceCode,
            "photon.glsl": "" if lamSource is None else lamSource.sourceCode,
            "ray.glsl": ray.sourceCode,
            "rng.glsl": "" if rng is None else rng.sourceCode,
            **materials.header,
        }
        code = compileShader("lightsource/sample/forward.glsl", "", headers)
        self._program = hp.Program(code)
        # bind memory
        materials.bindParams(self._program)

    @property
    def batchSize(self) -> int:
        """Number of samples drawn per run"""
        return self._batchSize

    @property
    def materials(self) -> MaterialStore | None:
        """Optional material store in use"""
        return self._materials

    @property
    def queue(self) -> IOQueue:
        """Queue holding the sampled light rays"""
        return self._queue

    @property
    def rayModel(self) -> RayModel:
        """Ray model used for sampling"""
        return self._ray

    @property
    def rng(self) -> RNG | None:
        """The random number generator used for sampling. None, if none used."""
        return self._rng

    @property
    def source(self) -> LightSource:
        """Light source that is sampled"""
        return self._source

    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all stages involved with this sampler in the correct
        order suitable for creating a pipeline.
        """
        stages: list[PipelineStage] = []
        if self.rng is not None:
            stages.append(self.rng)
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
            *self._queue.run(i),
        ]


class HostLightSource(LightSource):
    """
    Light source passing samples from the CPU to the GPU

    Parameters
    ----------
    capacity: int
        Maximum number of samples that can be drawn per run
    ray: RayModel | type[Structure]:
        Model describing the structure of rays in the queue
    updateFn: (HostLightSource, int) -> None | None, default=None
        Optional update function called before the pipeline processes a task.
        `i` is the i-th configuration the update should affect.
        Can be used to stream in new samples on demand.
    extra: set[str], default={}
        Set of extra parameter name, that can be set and retrieved using the
        stage api.
    """

    name = "Host Light Source"

    class LightParams(Structure):
        _fields_ = [("_queueAdr", buffer_reference), ("_queueSize", c_uint32)]

    def __init__(
        self,
        capacity: int,
        ray: RayModel | type[Structure],
        *,
        updateFn: Callable[[HostLightSource, int], None] | None = None,
        extra: set[str] = set(),
    ) -> None:
        super().__init__(
            params={"LightParams": HostLightSource.LightParams},
            extra=extra,
        )

        if isinstance(ray, RayModel):
            ray = ray.forwardItem
        # allocate queue
        self._queue = IOQueue(ray, capacity, mode="update", skipCounter=True)
        assert self._queue.tensor is not None
        # save params
        self._capacity = capacity
        self._updateFn = updateFn
        self.setParams(_queueAdr=self._queue.tensor.address, _queueSize=capacity)

    @property
    def queue(self) -> IOQueue:
        """Queue holding the samples for the next batch"""
        return self._queue

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("lightsource/host.glsl")

    def update(self, i: int) -> None:
        if self._updateFn is not None:
            self._updateFn(self, i)
        super().update(i)

    def run(self, i: int) -> list[hp.Command]:
        return [*self.queue.run(i), *super().run(i)]


class StreamingHostLightSource(HostLightSource):
    """
    Light source using a provided data source to sequentially feed light rays to
    the device. Subsequent batches move forward into the provided data streaming
    it to the device. Providing new data resets the stream.

    Parameters
    ----------
    capacity: int
        Maximum number of rays that can be drawn per run
    ray: RayModel | type[Structure]
        Ray model or structure describing the data in the queue to be fed to the
        GPU
    data: dict[str, ArrayLike] | None, default=None
        Data source from which the batches are produced.

    Stage Parameters
    ----------------
    data: dict[str, ArrayLike] | None
        Data source from which the batches are produced.

    Note
    ----
    There is no special handling for when the last batch from the data is not
    filling the queue completely. Data in the queue beyond the end is likely old
    or garbage, but the source does not prevent the tracer from reading it.
    """

    def __init__(
        self,
        capacity: int,
        ray: RayModel | type[Structure],
        *,
        data: dict[str, ArrayLike] | None = None,
    ) -> None:
        super().__init__(capacity, ray, updateFn=self._updateData, extra={"data"})
        self._dataFields = set(self.queue.view(0).fields)
        self.data = data

    @property
    def data(self) -> dict[str, NDArray] | None:
        """Data used as streaming source"""
        return self._data

    @data.setter
    def data(self, value: dict[str, ArrayLike] | None) -> None:
        self._offset = 0
        self._dataSize = 0
        if value is None:
            self._data = None
            return
        # check if all the needed fields are there
        if not self._dataFields.issubset(value.keys()):
            m = self._dataFields.difference(value.keys())
            raise ValueError(f"provided data misses expected fields: {m}")
        self._dataSize = np.size(value["wavelength"], 0)
        # ensure all fields have the same size
        for field in self._dataFields:
            if np.size(value[field], 0) != self._dataSize:
                raise ValueError("fields of provided data differ in size!")
            # TODO: Check correct dimensions, too?
        # create ndarray and store them for later
        self._data = {f: np.asarray(value[f]) for f in self._dataFields}

    @data.deleter
    def data(self) -> None:
        self.data = None

    @staticmethod
    def _updateData(source: HostLightSource, i: int) -> None:
        assert isinstance(source, StreamingHostLightSource)
        if source.data is None:
            warn(
                "StreamingHostLightSource: "
                "New data was requested but no source was given!"
            )
            return
        if source._offset >= source._dataSize:
            warn("StreamingHostLightSource ran out of data!")
            return

        # calculate next data range
        start = source._offset
        end = min(source._offset + source._capacity, source._dataSize)
        count = end - start
        source._offset = end
        # copy new data block
        view = source.queue.view(i)
        for field in source._dataFields:
            view[field][:count] = source.data[field][start:end]


class ConeLightSource(LightSource):
    """
    Point light source radiating light in a specified cone.
    Can optionally be constant polarized.

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from. Required for forward mode.
    mediumIdx: int
        Index of the medium the light source is located in
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Position of the light source.
    direction: (float, float, float), default(1.0, 0.0, 0.0)
        Direction of the opening cone.
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    cosOpeningAngle: float, default=0.5
        Cosine of the cones opening angle
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons per wavelength. Ignored if source emits particles.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If so, `budget` will be
        ignored.
    """

    name = "Cone Light Source"

    class LightParams(Structure):
        _fields_ = [
            ("direction", vec3),
            ("cosOpeningAngle", c_float),
            ("position", vec3),
            ("mediumIdx", c_uint32),
            ("_contribFwd", c_float),
            ("_contribBwd", c_float),
            ("timeRange", vec2),
        ]

    def __init__(
        self,
        wavelengthSource: WavelengthSource | None = None,
        *,
        mediumIdx: int,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
        timeRange: tuple[float, float] = (0.0, 100.0) * u.ns,
        cosOpeningAngle: float = 0.5,
        budget: float = 1.0,
        emitParticles: bool = False,
    ) -> None:
        lam = wavelengthSource
        super().__init__(
            nRNGForward=0 if lam is None else lam.nRNGSamples + 3,
            nRNGBackward=1,
            wavelengthSource=wavelengthSource,
            params={"LightParams": ConeLightSource.LightParams},
            extra={"budget"},
        )
        if emitParticles and budget != 1.0:
            warn("Cannot set budget: Light source emits particles")
        # save params
        self._emitParticles = emitParticles
        self._budget = budget
        self.setParams(
            position=position,
            direction=direction,
            timeRange=timeRange,
            cosOpeningAngle=cosOpeningAngle,
            mediumIdx=mediumIdx,
        )

    @property
    def budget(self) -> float:
        """
        Total energy or amount of photons the light source distributes among all
        sampled rays per wavelength. Ignored if source emits particles.
        """
        return self._budget

    @budget.setter
    def budget(self, value: float) -> None:
        if self.emitsParticles:
            warn("Cannot set budget: Light source emits particles")
        self._budget = value

    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def backwardSourceCode(self) -> str | None:
        if self.emitsParticles:
            return None
        else:
            return loadShader("lightsource/cone/backward.glsl")

    @property
    def forwardSourceCode(self) -> str | None:
        if self.wavelengthSource is None:
            return None
        else:
            preamble = createPreamble(LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles)
            code = loadShader("lightsource/cone/forward.glsl")
            return preamble + code

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        c = self.budget
        self.setParam("_contribFwd", c)
        # in forward parameter volume cancels with the probability
        # This is not the case in backward mode
        # -> divide contrib by parameter space volume
        c /= 2.0 * np.pi * (1.0 - self.cosOpeningAngle)
        self.setParam("_contribBwd", c)


class PencilLightSource(LightSource):
    """
    Light source generating a pencil beam

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from
    mediumIdx: int
        Index of the medium the light source is located in
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Start point of the ray
    direction: (float, float, float), default=(0.0, 0.0, 1.0)
        Direction of the ray
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons per wavelength. Ignored if source emits particles.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If so, `budget` will be
        ignored.
    """

    name = "Pencil Light Source"

    class LightParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("direction", vec3),
            ("_budget", c_float),
            ("mediumIdx", c_uint32),
            ("timeRange", vec2),
        ]

    def __init__(
        self,
        wavelengthSource: WavelengthSource,
        *,
        mediumIdx: int,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        timeRange: tuple[float, float] = (0.0, 100.0) * u.ns,
        budget: float = 1.0,
        emitParticles: bool = False,
    ) -> None:
        super().__init__(
            nRNGForward=wavelengthSource.nRNGSamples + 1,
            wavelengthSource=wavelengthSource,
            params={"LightParams": PencilLightSource.LightParams},
            extra={"budget"},
        )
        if emitParticles and budget != 1.0:
            warn("Cannot set budget: Light source emits particles")
        # save params
        self._emitParticles = emitParticles
        self._budget = budget
        self.setParams(
            position=position,
            direction=direction,
            timeRange=timeRange,
            mediumIdx=mediumIdx,
        )

    @property
    def budget(self) -> float:
        """
        Total energy or amount of photons the light source distributes among all
        sampled rays per wavelength. Ignored if source emits particles.
        """
        return self._budget

    @budget.setter
    def budget(self, value: float) -> None:
        if self.emitsParticles:
            warn("Cannot set budget: Light source emits particles")
        self._budget = value

    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def forwardSourceCode(self) -> str:
        preamble = createPreamble(LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles)
        code = loadShader("lightsource/pencil.glsl")
        return preamble + code


class SphericalLightSource(LightSource):
    """
    Isotropic point light source radiating light in all directions uniformly.

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from. Required for forward mode.
    mediumIdx: int
        Index of the medium the light source is located in
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Position of the light source.
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons per wavelength. Ignored if source emits particles.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If so, `budget` will be
        ignored.
    """

    class LightParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("mediumIdx", c_uint32),
            ("timeRange", vec2),
            ("_contribFwd", c_float),
            ("_contribBwd", c_float),
        ]

    def __init__(
        self,
        wavelengthSource: WavelengthSource | None = None,
        *,
        mediumIdx: int,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        timeRange: tuple[float, float] = (0.0, 100.0) * u.ns,
        budget: float = 1.0,
        emitParticles: bool = False,
    ) -> None:
        lam = wavelengthSource
        super().__init__(
            nRNGForward=0 if lam is None else lam.nRNGSamples + 3,
            nRNGBackward=1,
            wavelengthSource=wavelengthSource,
            params={"LightParams": SphericalLightSource.LightParams},
            extra={"budget"},
        )
        if emitParticles and budget != 1.0:
            warn("Cannot set budget: Light source emits particles")
        # save params
        self._emitParticles = emitParticles
        self._budget = budget
        self.setParams(
            position=position,
            mediumIdx=mediumIdx,
            timeRange=timeRange,
        )

    @property
    def budget(self) -> float:
        """
        Total energy or amount of photons the light source distributes among all
        sampled rays per wavelength. Ignored if source emits particles.
        """
        return self._budget

    @budget.setter
    def budget(self, value: float) -> None:
        if self.emitsParticles:
            warn("Cannot set budget: Light source emits particles")
        self._budget = value

    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def forwardSourceCode(self) -> str | None:
        if self.wavelengthSource is None:
            return None
        else:
            preamble = createPreamble(LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles)
            code = loadShader("lightsource/spherical/forward.glsl")
            return preamble + code

    @property
    def backwardSourceCode(self) -> str | None:
        if self.emitsParticles:
            return None
        else:
            return loadShader("lightsource/spherical/backward.glsl")

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        c = self.budget
        self.setParam("_contribFwd", c)
        # in forward parameter volume cancels with the probability
        # This is not the case in backward mode
        # -> divide contrib by parameter space volume
        c /= 4.0 * np.pi
        self.setParam("_contribBwd", c)
