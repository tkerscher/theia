from __future__ import annotations

from abc import abstractmethod

import hephaistos as hp
from hephaistos import Program
from hephaistos.pipeline import PipelineStage, SourceCodeMixin

from ctypes import Structure, c_float, c_int32, c_uint32, c_uint64
from ctypes import addressof, memset, sizeof
from hephaistos.glsl import vec3
from numpy.ctypeslib import as_array

from theia.camera import Camera
from theia.compiler import createPreamble, compileShader, loadShader
from theia.light import LightSource, WavelengthSource
from theia.material import MaterialStore
from theia.random import RNG
from theia.ray import RayModel
from theia.response import HitResponse, TraceConfig
from theia.scene import RectBBox
from theia.target import Target
from theia.volume import VolumeModel
import theia.units as u

from enum import IntEnum
from numpy.typing import NDArray


__all__ = [
    "EmptyEventCallback",
    "EventResultCode",
    "EventStatisticCallback",
    "TraceEventCallback",
    "Tracer",
    "TrackRecordCallback",
    "VolumeForwardTracer",
]


def __dir__():
    return __all__


class TraceEventCallback(SourceCodeMixin):
    """
    Base class for callbacks a tracer will call on tracing events
    (scatter, hit, detected, lost, decayed) of the form

    void onEvent(const Ray ray, uint type, uint idx, uint i)
    """

    name = "Trace Event Callback"

    def __init__(
        self,
        *,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
    ) -> None:
        super().__init__(params, extra)


class EmptyEventCallback(TraceEventCallback):
    """Callback ignoring all events"""

    def __init__(self) -> None:
        super().__init__()

    @property
    def sourceCode(self) -> str:
        return loadShader("callback/empty.glsl")


class EventStatisticCallback(TraceEventCallback):
    """
    Callback recording statistics about the event types, i.e. increments a
    global counter for each event type. Current statistic are read directly from
    the device, but may have a bad impact on currently running tracing.
    """

    name = "Event Statistic"

    class Statistic(Structure):
        _fields_ = [
            ("created", c_uint64),
            ("scattered", c_uint64),
            ("hit", c_uint64),
            ("detected", c_uint64),
            ("volume", c_uint64),
            ("lost", c_uint64),
            ("decayed", c_uint64),
            ("absorbed", c_uint64),
            ("missed", c_uint64),
            ("maxIter", c_uint64),
            ("error", c_uint64),
            ("mismatch", c_uint64),
        ]

    def __init__(self) -> None:
        super().__init__()
        # allocate mapped tensor for statistics
        self._tensor = hp.Tensor(sizeof(self.Statistic), mapped=True)
        if not self._tensor.isMapped:
            raise RuntimeError("Could not create mapped tensor")
        self._stat = self.Statistic.from_address(self._tensor.memory)
        self.reset()

    @property
    def absorbed(self) -> int:
        """Number of absorbed rays"""
        return self._stat.absorbed

    @property
    def created(self) -> int:
        """Number of rays created"""
        return self._stat.created

    @property
    def hit(self) -> int:
        """Number of hit events"""
        return self._stat.hit

    @property
    def detected(self) -> int:
        """Number of rays detected"""
        return self._stat.detected

    @property
    def maxIter(self) -> int:
        """Number of rays that reached the maximum path length"""
        return self._stat.maxIter

    @property
    def mismatch(self) -> int:
        """Number of rays aborted due to hitting unexpected media"""
        return self._stat.mismatch

    @property
    def missed(self) -> int:
        """Number of rays that missed the target"""
        return self._stat.missed

    @property
    def scattered(self) -> int:
        """Number of scatter events"""
        return self._stat.scattered

    @property
    def lost(self) -> int:
        """Number of rays that left the tracing boundary box"""
        return self._stat.lost

    @property
    def decayed(self) -> int:
        """Number of rays exceeding the max time"""
        return self._stat.decayed

    @property
    def volume(self) -> int:
        """Number of volume boundary hits"""
        return self._stat.volume

    @property
    def error(self) -> int:
        """Number of traces aborted due to an error"""
        return self._stat.error

    @property
    def sourceCode(self) -> str:
        return loadShader("callback/stat.glsl")

    def reset(self) -> None:
        """Resets the statistic"""
        memset(addressof(self._stat), 0, sizeof(self.Statistic))

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        program.bindParams(Statistics=self._tensor)

    def __str__(self) -> str:
        fields = sorted([f[0] for f in self.Statistic._fields_])
        just = max(len(f) for f in fields)
        lines = [f"{f.rjust(just)}: {getattr(self, f):,}" for f in fields]
        return "\n".join(lines)


class TrackRecordCallback(TraceEventCallback):
    """
    Callback recording position of events allowing to later reconstruct complete
    path. Only considers first sample for time coordinate.

    Parameters
    ----------
    capacity: int
        Number of rays that can be recorded
    length: int
        Max number of points per track
    retrieve: bool, default=True
        Whether to retrieve the tracks from the device
    """

    name = "Track Recorder"

    def __init__(
        self,
        capacity: int,
        length: int,
        *,
        retrieve: bool = True,
    ) -> None:
        super().__init__()
        # save params
        self._capacity = capacity
        self._length = length
        self._retrieve = retrieve

        # allocate memory
        words = (1 + 5 * length) * capacity
        self._tensor = hp.Tensor(words * 4)
        self._buffer = [hp.Buffer(words * 4) for _ in range(2)]

    @property
    def capacity(self) -> int:
        """Number of rays that can be recorded"""
        return self._capacity

    @property
    def length(self) -> int:
        """Max number of points per track"""
        return self._length

    @property
    def polarized(self) -> bool:
        """Whether polarization state is recorded"""
        return self._polarized

    @property
    def retrieve(self) -> bool:
        """Wether to retrieve the tracks from the device"""
        return self._retrieve

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            TRACK_COUNT=self.capacity,
            TRACK_LENGTH=self.length,
        )
        code = loadShader("callback/track.glsl")
        return preamble + code

    @property
    def tensor(self) -> hp.Tensor:
        """Tensor containing the tracks"""
        return self._tensor

    def result(self, i: int) -> tuple[NDArray, NDArray, NDArray]:
        """
        Returns the recorded tracks saved using the i-th pipeline configuration.
        First tuple are the tracks, the second one the length of each track, and
        the last one is the last recorded result code. Positions after each
        track length may contain garbage data.

        Returns
        -------
        tracks: NDArray
            Recorded tracks stored in a numpy array of shape (track,pos,4)
        lengths: NDArray
            Length of each recorded track stored in a numpy array of shape (track,)
        codes: NDArray
            Recorded result codes for each track point of shape (track,pos)
        """
        # fetch each data structures
        adr = self._buffer[i].address
        pLengths = (c_uint32 * self.capacity).from_address(adr)
        adr += 4 * self.capacity
        n = self.length * self.capacity
        pTracks = (c_float * (4 * n)).from_address(adr)
        adr += 4 * 4 * n
        pCodes = (c_int32 * n).from_address(adr)
        # construct numpy arrays
        lengths = as_array(pLengths)
        tracks = as_array(pTracks).reshape((4, self.length, self.capacity))
        codes = as_array(pCodes).reshape((self.length, self.capacity))
        # change from (coords, ray, event) -> (ray, event, coords)
        tracks = tracks.transpose(2, 1, 0)
        codes = codes.transpose()
        return tracks, lengths, codes

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        program.bindParams(TrackBuffer=self._tensor)

    def run(self, i: int) -> list[hp.Command]:
        if self.retrieve:
            return [hp.retrieveTensor(self._tensor, self._buffer[i])]
        else:
            return []


class EventResultCode(IntEnum):
    """
    Enumeration of result codes the tracing algorithm can encounter and pass
    on to the event callback. Negative codes indicate the tracer to stop.
    """

    SUCCESS = 0
    """Operation successful without further information"""
    RAY_CREATED = 1
    """Ray was sampled"""
    RAY_SCATTERED = 2
    """Ray scattered"""
    RAY_HIT = 3
    """Ray hit a geometry other than target"""
    RAY_DETECTED = 4
    """Ray hit target"""
    VOLUME_HIT = 5
    """Ray hit volume border"""
    RAY_LOST = -1
    """Ray left tracing boundary box"""
    RAY_DECAYED = -2
    """Ray reached max life time"""
    RAY_ABSORBED = -3
    """Ray hit absorber"""
    RAY_MISSED = -4
    """Ray meant to create a hit missed target"""
    MAX_ITER = -5
    """Tracing reached max iteration"""
    ERROR_CODE_MAX_VALUE = -10
    """Max value for an error code"""
    ERROR_UNKNOWN = -10
    """Tracer encountered an unexpected error indicating a bug in the tracer"""
    ERROR_MEDIA_MISMATCH = -11
    """Tracer encountered unexpected media"""
    ERROR_TRACE_ABORT = -12
    """Tracer reached a state it can't proceed from"""


class Tracer(PipelineStage):
    """
    Base class for tracing algorithms sampling light paths connecting source
    and target.
    """

    name = "Tracer"

    def __init__(
        self,
        batchSize: int,
        ray: RayModel,
        response: HitResponse,
        rng: RNG,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
        *,
        capacity: int | None,
        callback: TraceEventCallback,
        maxPathLength: int,
        nRNGSamples: int,
        maxHitsPerRay: int,
    ) -> None:
        super().__init__(params, extra)

        if maxPathLength < 1:
            raise ValueError("Max path length must be at least 1")

        self._ray = ray
        self._response = response
        self._rng = rng
        self._capacity = batchSize if capacity is None else capacity
        self._callback = callback
        self._maxPathLength = maxPathLength
        self._nRNGSamples = nRNGSamples
        self._maxHitsPerRay = maxHitsPerRay
        self.batchSize = batchSize
        response.prepare(self._getTraceConfig(), ray)

    @property
    def batchSize(self) -> int:
        """Number of rays to simulate per run"""
        return self._batchSize

    @batchSize.setter
    def batchSize(self, value: int) -> None:
        if value < 0:
            raise ValueError("batchSize must be at least zero!")
        if value > self.capacity:
            raise ValueError("batchSize cannot be larger than capacity!")
        self._batchSize = value

    @property
    def capacity(self) -> int:
        """Maximal batch size"""
        return self._capacity

    @property
    def callback(self) -> TraceEventCallback:
        """Callback called for each tracing event"""
        return self._callback

    @property
    def maxHits(self) -> int:
        """Maximum number of hits per batch"""
        return self.maxHitsPerRay * self.batchSize

    @property
    def maxHitsPerRay(self) -> int:
        """Maximum amount of hits the tracer can produce per ray"""
        return self._maxHitsPerRay

    @property
    def maxPathLength(self) -> int:
        """Maximum length of ray paths sampled"""
        return self._maxPathLength

    @property
    def normalization(self) -> float:
        """
        Normalization factor that must be applied to each sample to get a
        correct estimate.
        """
        return 1.0 if self.ray.isParticle else 1.0 / self.batchSize

    @property
    def nRNGSamples(self) -> int:
        """Amount of random numbers drawn per ray"""
        return self._nRNGSamples

    @property
    def ray(self) -> RayModel:
        """Physical model of the rays to be traced"""
        return self._ray

    @property
    def response(self) -> HitResponse:
        """Response function processing each simulated hit"""
        return self._response

    @property
    def rng(self) -> RNG:
        """Random number generator used"""
        return self._rng

    def collectStages(
        self,
        *,
        useOriginalNames: bool = False,
    ) -> list[tuple[str, PipelineStage]]:
        """
        Returns a list of all pipeline stages involved with this tracer in the
        correct order suitable for creating a pipeline. If useOriginalNames is
        False, stages will be named according to their function in the tracer,
        instead of their original names.
        """
        stages = self._collectStages()
        if useOriginalNames:
            return [(s.name, s) for _, s in stages]
        else:
            return stages

    @abstractmethod
    def _collectStages(self) -> list[tuple[str, PipelineStage]]:
        """
        Internal implementation of collectStages() in derived classes returning
        the list of renamed pipeline stages in correct order.
        """
        pass

    def _getTraceConfig(self) -> TraceConfig:
        return TraceConfig(
            self.batchSize,
            self.capacity,
            self.maxHitsPerRay,
            self.normalization,
        )

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        # give response a chance to respond to any changes
        self.response.updateConfig(self._getTraceConfig())


def _getVolumeForwardRNGStride(
    volume: VolumeModel,
    response: HitResponse,
    target: Target | None = None,
) -> int:
    """util func calculating number of rng draws per volume tracing"""
    result = 0
    result += max(1, volume.rngDraws.sampleInteractionLength)
    result += volume.rngDraws.applyVolumeEffect
    result += volume.rngDraws.sampleVolumeInteraction
    result += response.nRNGSamples
    if target is not None:
        result += volume.rngDraws.sampleVolumeScattering
        result += target.nRNGSamples
        result += 2 * volume.rngDraws.applyVolumeEffect
        result += 2 * volume.rngDraws.volumeScatterRay
        result += response.nRNGSamples
    return result


def _getVolumeBackwardRNGStride(
    volume: VolumeModel,
    response: HitResponse,
    target: Target | None = None,
) -> int:
    result = 0
    result += max(1, volume.rngDraws.sampleInteractionLength)
    result += volume.rngDraws.applyVolumeEffect
    result += volume.rngDraws.sampleVolumeInteraction
    result += response.nRNGSamples
    if target is not None:
        result += target.nRNGSamples
        result += volume.rngDraws.volumeScatterRay
        result += volume.rngDraws.applyVolumeEffect
        result += response.nRNGSamples
    return result


class VolumeForwardTracer(Tracer):
    """
    Path tracer simulating rays traveling through media originating at a source
    and ending at a detector. It does not contain a full scene, but only a
    single target. Can be run on hardware without ray tracing support and may be
    faster.

    Parameters
    ----------
    batchSize: int
        Number of rays to simulate per run.
    ray: RayModel
        Model of the ray to propagate
    source: LightSource
        Source producing rays
    target: Target
        Target model used to generate hits
    response: HitResponse
        Respose function processing each simulated hit
    rng: RNG
        Random number generator
    materials: MaterialStore
        MaterialStore containing the medium used by the tracer
    volume: VolumeModel | str | int
        Volume model used by the tracer. Can dynamically be fetched from the
        material store using the corresponding medium's name or index.
    capacity: int | None, default=None
        Maximum batch size. If None, same as `batchSize`.
    callback: TraceEventCallback, default=EmptyEventCallback()
        Callback called for each tracing event. See `TraceEventCallback`
    maxPathLength: int, default=64
        Maximum length of paths to simulate
    objectId: int, default=0
        Object id to be used for creating hits.
    sampleCoefficient: float, default=NaN
        Optional coefficient used for sampling ray lengths. If the value is NaN
        or negative, the volume will sample the ray length instead. Will always
        be disabled, if the ray is a particle.
    maxTime: float, default=inf
        Rays exceeding this limit will aborted with code `decayed`.
    traceBBox: RectBBox | None, default=None
        Rays leaving the trace boundary box will get stopped and noted as lost.
        If `None`, this defaults to a box with at the origin spanning 1km in
        each direction, i.e. 2km side length.
    disableDirectLighting: bool, default=False
        Whether to ignore contributions from direct lighting, i.e. paths with
        no scattering. Usefull if a dedicated tracer or estimator for direct
        light contributions is additionally used.
    disableTargetSampling: bool, default=False
        Whether to sample the target after scattering events to create hits.
        Disabling it means that targets will only be hit by chance.
        Automatically disabled if the ray is a particle.

    Stage Parameters
    ----------------
    batchSize: int
        Number of rays to simulate per run.
    objectId: int
        Object id to use for creating hits.
    maxTime: float
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    """

    class TraceParams(Structure):
        _fields_ = [
            ("_batchSize", c_uint32),
            ("_mediumIdx", c_uint32),
            ("objectId", c_int32),
            ("sampleCoefficient", c_float),
            ("lowerBBoxCorner", vec3),
            ("upperBBoxCorner", vec3),
            ("maxTime", c_float),
            ("_maxDist", c_float),
        ]

    def __init__(
        self,
        batchSize: int,
        ray: RayModel,
        source: LightSource,
        target: Target,
        response: HitResponse,
        rng: RNG,
        materials: MaterialStore,
        *,
        volume: VolumeModel | str | int,
        capacity: int | None = None,
        callback: TraceEventCallback = EmptyEventCallback(),
        maxPathLength: int = 64,
        objectId: int = 0,
        sampleCoefficient: float = float("NaN"),
        maxTime: float = float("inf"),
        traceBBox: RectBBox | None = None,
        disableDirectLighting: bool = False,
        disableTargetSampling: bool = False,
    ) -> None:
        # enforce params in particle mode
        if ray.isParticle:
            disableTargetSampling = True
        # validate params compatability
        if not isinstance(volume, VolumeModel):
            volume = materials.getVolumeModel(volume)
        if volume.forwardSourceCode is None:
            raise ValueError(
                f'Volume model "{volume.name}" does not support forward tracing'
            )
        if not ray.supportsForward:
            raise ValueError("Ray model does not support forward tracing")
        if ray.isParticle and not volume.supportsParticle:
            raise ValueError("Volume has no particle support as required by ray")
        if not source.supportForward:
            raise ValueError("Light source does not support forward mode")
        if not disableTargetSampling and not volume.supportsNEE:
            raise ValueError("Volume supports no target sampling")
        # calculate max hits per ray
        if not disableTargetSampling:
            maxHitsPerRay = 2 * (maxPathLength - 1)
            if not disableDirectLighting:
                maxHitsPerRay += 1
        else:
            maxHitsPerRay = 1
        # NEE extends path length by one -> reduce path length by one
        pathLength = maxPathLength if disableTargetSampling else maxPathLength - 1
        # calculate amount of random numbers needed
        rngInit = source.nRNGForward
        if not disableDirectLighting:
            rngInit += volume.rngDraws.applyVolumeEffect
            rngInit += response.nRNGSamples
        rngStride = _getVolumeForwardRNGStride(volume, response, target)
        nRNG = rngInit + pathLength * rngStride
        super().__init__(
            batchSize,
            ray,
            response,
            rng,
            {"TraceParams": VolumeForwardTracer.TraceParams},
            capacity=capacity,
            callback=callback,
            maxPathLength=maxPathLength,
            nRNGSamples=nRNG,
            maxHitsPerRay=maxHitsPerRay,
        )
        # save params
        if traceBBox is None:
            traceBBox = RectBBox((-1.0, -1.0, -1.0) * u.km, (1.0, 1.0, 1.0) * u.km)
        self._source = source
        self._target = target
        self._materials = materials
        self._directLightingDisabled = disableDirectLighting
        self._targetSamplingDisabled = disableTargetSampling
        self.setParams(
            sampleCoefficient=sampleCoefficient,
            objectId=objectId,
            maxTime=maxTime,
            lowerBBoxCorner=traceBBox.lowerCorner,
            upperBBoxCorner=traceBBox.upperCorner,
            _maxDist=traceBBox.diagonal,
        )
        # compile code
        preamble = createPreamble(
            DISABLE_DIRECT_LIGHTING=disableDirectLighting,
            DISABLE_NEE=disableTargetSampling,
            PATH_LENGTH=pathLength,
        )
        photons = source.wavelengthSource
        headers = {
            "callback.glsl": callback.sourceCode,
            "photon.glsl": "" if photons is None else photons.sourceCode,
            "ray.glsl": ray.sourceCode,
            "response.glsl": response.sourceCode,
            "rng.glsl": rng.sourceCode,
            "source.glsl": source.sourceCode,
            "target.glsl": target.sourceCode,
            "volume.glsl": volume.forwardSourceCode,
            **materials.header,
        }
        code = compileShader("tracer/volume/forward.glsl", preamble, headers)
        # create program
        self._program = hp.Program(code)
        materials.bindParams(self._program)

    @Tracer.batchSize.setter
    def batchSize(self, value) -> None:
        assert Tracer.batchSize.fset is not None  # to make linter happy
        Tracer.batchSize.fset(self, value)
        self.setParam("_batchSize", value)

    @property
    def directLightingDisabled(self) -> bool:
        """Whether direct lighting is disabled"""
        return self._directLightingDisabled

    @property
    def materials(self) -> MaterialStore:
        """MaterialStore containing the medium used by the tracer"""
        return self._materials

    @property
    def source(self) -> LightSource:
        """Source producing rays"""
        return self._source

    @property
    def target(self) -> Target:
        """Target model used to determine hits"""
        return self._target

    @property
    def targetSamplingDisabled(self) -> bool:
        """Whether target sampling is disabled"""
        return self._targetSamplingDisabled

    @property
    def traceBBox(self) -> RectBBox:
        """Boundary box of simulated rays"""
        return RectBBox(
            self.getParam("lowerBBoxCorner"), self.getParam("upperBBoxCorner")
        )

    def _collectStages(self) -> list[tuple[str, PipelineStage]]:
        stages = [
            ("rng", self.rng),
            # ("photons", self.wavelengthSource),
            ("lightSource", self.source),
            ("target", self.target),
            ("tracer", self),
            ("callback", self.callback),
            ("response", self.response),
        ]
        # wavelength sources are optional, but need to be before the light source
        if self.source.wavelengthSource is not None:
            stages.insert(1, ("photons", self.source.wavelengthSource))
        return stages

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        self.setParam("_maxDist", self.traceBBox.diagonal)

    def run(self, i: int) -> list[hp.Command]:
        for _, stage in self.collectStages():
            stage.bindParams(self._program, i)
        groups = -(self.capacity // -512)
        return [self._program.dispatch(groups)]


class VolumeBackwardTracer(Tracer):
    """
    Path tracer sampling paths starting at a camera and creating complete ones
    by sampling the light source independently at each volume interaction. It
    does NOT check for any intersection with a scene but may include self
    shadowing from an optional target acting as a proxy for the physical camera.
    Can be run on hardware without ray tracing support and may be faster.

    Parameters
    ----------
    batchSize: int
        Number of rays to simulate per run. Note that a single ray may generate
        up to nScattering hits plus one if direct lighting is enabled.
    ray: RayModel
        Model of the ray to propagate
    source: LightSource
        Source producing light rays. Must support backward mode.
    camera: Camera
        Source producing camera rays. If direct lighting is enabled, must
        support direct mode.
    wavelengthSource: WavelengthSource
        Source producing wavelength samples
    response: HitResponse
        Respose function processing each simulated hit
    rng: RNG
        Random number generator
    materials: MaterialStore
        MaterialStore containing the medium used by the tracer
    volume: VolumeModel | str | int
        Volume model used by the tracer. Can dynamically be fetched from the
        material store using the corresponding medium's name or index.
    capacity: int | None, default=None
        Maximum batch size. If None, same as `batchSize`.
    callback: TraceEventCallback, default=EmptyEventCallback()
        Callback called for each tracing event. See `TraceEventCallback`
    maxPathLength: int, default=64
        Maximum length of paths to simulate
    target: Target | None, default=None
        Optional target acting as proxy for the camera in determining self
        shadowing
    sampleCoefficient: float, default=NaN
        Optional coefficient used for sampling ray lengths. If the value is NaN
        or negative, the volume will sample the ray length instead. Will always
        be disabled, if the ray is a particle.
    maxTime: float, default=inf
        Rays exceeding this limit will aborted with code `decayed`.
    traceBBox: RectBBox | None, default=None
        Rays leaving the trace boundary box will get stopped and noted as lost.
        If `None`, this defaults to a box with at the origin spanning 1km in
        each direction, i.e. 2km side length.
    disableDirectLighting: bool, default=False
        Whether to ignore contributions from direct lighting, i.e. paths with
        no scattering. Usefull if a dedicated tracer or estimator for direct
        light contributions is additionally used.

    Stage Parameters
    ----------------
    scatterCoefficient: float
        Scatter coefficient used for sampling ray lengths. Negative values or
        NaN will cause the tracer to use the current scattering length
        instead. A value of zero disables volume scattering all together.
    medium: int | str
        Index or name of the medium the scene is emerged in. If given as a
        string, the index will be fetched from the given `MaterialStore`.
    lowerBBoxCorner: (float, float, float)
        Lower limit of the x,y,z coordinates a ray must stay above to not get
        stopped
    upperBBoxCorner: (float, float, float)
        Upper limit of the x,y,z coordinates a ray must stay below to not get
        stopped
    maxTime: float
        Max total time including delay from the source and travel time, after
        which a ray gets stopped

    Note
    ----
    No hits will be created if scattering is disabled.
    """

    name = "Volume Backward Tracer"

    class TraceParams(Structure):
        _fields_ = [
            ("_batchSize", c_uint32),
            ("sampleCoefficient", c_float),
            ("lowerBBoxCorner", vec3),
            ("upperBBoxCorner", vec3),
            ("maxTime", c_float),
            ("_maxDist", c_float),
        ]

    def __init__(
        self,
        batchSize: int,
        ray: RayModel,
        source: LightSource,
        camera: Camera,
        wavelengthSource: WavelengthSource,
        response: HitResponse,
        rng: RNG,
        materials: MaterialStore,
        *,
        volume: VolumeModel | str | int,
        capacity: int | None = None,
        callback: TraceEventCallback = EmptyEventCallback(),
        maxPathLength: int = 64,
        target: Target | None = None,
        sampleCoefficient: float = float("NaN"),
        maxTime: float = float("inf"),
        traceBBox: RectBBox | None = None,
        disableDirectLighting: bool = False,
    ) -> None:
        # fetch volume model if not given
        if not isinstance(volume, VolumeModel):
            volume = materials.getVolumeModel(volume)
        # check components support backward tracing
        if not ray.supportsBackward:
            raise ValueError("Ray model does not support backward tracing")
        if not source.supportBackward:
            raise ValueError("Light source does not support backward mode")
        if not disableDirectLighting and not camera.supportDirect:
            raise ValueError("Camera does not support direct mode")
        if volume.backwardSourceCode is None:
            raise ValueError("Volume model does not support backward mode")

        # calculate max hits
        maxHitsPerRay = maxPathLength - 1 if disableDirectLighting else maxPathLength
        # calculate number of rng samples
        rngInit = wavelengthSource.nRNGSamples + camera.nRNGSamples
        rngStride = _getVolumeBackwardRNGStride(volume, response, target)
        nRNG = rngInit + maxPathLength * rngStride
        if not disableDirectLighting:
            nRNG += camera.nRNGDirect
            nRNG += source.nRNGBackward
            nRNG += response.nRNGSamples
        super().__init__(
            batchSize,
            ray,
            response,
            rng,
            {"TraceParams": VolumeBackwardTracer.TraceParams},
            capacity=capacity,
            callback=callback,
            maxPathLength=maxPathLength,
            nRNGSamples=nRNG,
            maxHitsPerRay=maxHitsPerRay,
        )
        # save params
        if traceBBox is None:
            traceBBox = RectBBox((-1.0, -1.0, -1.0) * u.km, (1.0, 1.0, 1.0) * u.km)
        self._photons = wavelengthSource
        self._camera = camera
        self._source = source
        self._target = target
        self._materials = materials
        self._directLightingDisabled = disableDirectLighting
        self.setParams(
            sampleCoefficient=sampleCoefficient,
            maxTime=maxTime,
            lowerBBoxCorner=traceBBox.lowerCorner,
            upperBBoxCorner=traceBBox.upperCorner,
            _maxDist=traceBBox.diagonal,
        )
        # compile code
        preamble = createPreamble(
            DISABLE_DIRECT_LIGHTING=disableDirectLighting,
            DISABLE_SELF_SHADOWING=target is None,
            PATH_LENGTH=maxPathLength - 1,
        )
        headers = {
            "callback.glsl": callback.sourceCode,
            "camera.glsl": camera.sourceCode,
            "photon.glsl": wavelengthSource.sourceCode,
            "ray.glsl": ray.sourceCode,
            "response.glsl": response.sourceCode,
            "rng.glsl": rng.sourceCode,
            "source.glsl": source.sourceCode,
            "target.glsl": "" if target is None else target.sourceCode,
            "volume.glsl": volume.backwardSourceCode,
            **materials.header,
        }
        code = compileShader("tracer/volume/backward.glsl", preamble, headers)
        # create program
        self._program = hp.Program(code)
        self.materials.bindParams(self._program)

    @Tracer.batchSize.setter
    def batchSize(self, value) -> None:
        assert Tracer.batchSize.fset is not None  # to make linter happy
        Tracer.batchSize.fset(self, value)
        self.setParam("_batchSize", value)

    @property
    def camera(self) -> Camera:
        """Camera producing camera rays"""
        return self._camera

    @property
    def directLightingDisabled(self) -> bool:
        """Whether direct lighting is disabled"""
        return self._directLightingDisabled

    @property
    def materials(self) -> MaterialStore:
        """MaterialStore containing the medium used by the tracer"""
        return self._materials

    @property
    def source(self) -> LightSource:
        """Source producing rays"""
        return self._source

    @property
    def target(self) -> Target | None:
        """Optional target used for self shadowing"""
        return self._target

    @property
    def traceBBox(self) -> RectBBox:
        """Boundary box of simulated rays"""
        return RectBBox(
            self.getParam("lowerBBoxCorner"), self.getParam("upperBBoxCorner")
        )

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Sampler for wavelengths"""
        return self._photons

    def _collectStages(self) -> list[tuple[str, PipelineStage]]:
        stages = [
            ("rng", self.rng),
            ("photons", self.wavelengthSource),
            ("source", self.source),
            ("camera", self.camera),
            ("tracer", self),
            ("callback", self.callback),
            ("response", self.response),
        ]
        if self.target is not None:
            stages.insert(4, ("target", self.target))
        return stages

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        self.setParam("_maxDist", self.traceBBox.diagonal)

    def run(self, i: int) -> list[hp.Command]:
        for _, stage in self.collectStages():
            stage.bindParams(self._program, i)
        groups = -(self.capacity // -512)
        return [self._program.dispatch(groups)]


class VolumeDirectTracer(Tracer):
    """
    Path tracer sampling both camera and light source to create direct
    connections between them. It does NOT check for any intersection with a
    scene but may include self shadowing from an optional target acting as a
    proxy for the physical camera. Can be run on hardware without ray tracing
    support and may be faster.

    Parameters
    ----------
    batchSize: int
        Number of rays to simulate per run.
    ray: RayModel
        Model of the ray to propagate
    source: LightSource
        Source producing light rays. Must support backward mode.
    camera: Camera
        Source producing camera rays. Must support direct mode.
    wavelengthSource: WavelengthSource
        Source producing wavelength samples
    response: HitResponse
        Respose function processing each simulated hit
    rng: RNG
        Random number generator
    materials: MaterialStore
        MaterialStore containing the medium used by the tracer
    volume: VolumeModel | str | int
        Volume model used by the tracer. Can dynamically be fetched from the
        material store using the corresponding medium's name or index.
    capacity: int | None, default=None
        Maximum batch size. If None, same as `batchSize`.
    callback: TraceEventCallback, default=EmptyEventCallback()
        Callback called for each tracing event. See `TraceEventCallback`
    target: Target | None, default=None
        Optional target acting as proxy for the camera in determining self
        shadowing
    maxTime: float, default=inf
        Rays exceeding this limit will aborted with code `decayed`.
    """

    name = "Volume Direct Tracer"

    class TraceParams(Structure):
        _fields_ = [
            ("maxTime", c_float),
            ("_batchSize", c_uint32),
        ]

    def __init__(
        self,
        batchSize: int,
        ray: RayModel,
        source: LightSource,
        camera: Camera,
        wavelengthSource: WavelengthSource,
        response: HitResponse,
        rng: RNG,
        materials: MaterialStore,
        *,
        volume: VolumeModel | str | int,
        capacity: int | None = None,
        callback: TraceEventCallback = EmptyEventCallback(),
        target: Target | None = None,
        maxTime: float = float("inf"),
    ) -> None:
        # fetch volume model if not given
        if not isinstance(volume, VolumeModel):
            volume = materials.getVolumeModel(volume)
        # check components support backward tracing
        if not ray.supportsBackward:
            raise ValueError("Ray model does not support backward tracing")
        if not source.supportBackward:
            raise ValueError("Light source does not support backward mode")
        if not camera.supportDirect:
            raise ValueError("Camera does not support direct mode")
        if volume.backwardSourceCode is None:
            raise ValueError("Volume model does not support backward mode")

        nRNG = (
            wavelengthSource.nRNGSamples
            + source.nRNGBackward
            + camera.nRNGDirect
            + volume.rngDraws.applyVolumeEffect
            + response.nRNGSamples
        )
        super().__init__(
            batchSize,
            ray,
            response,
            rng,
            {"TraceParams": VolumeDirectTracer.TraceParams},
            capacity=capacity,
            callback=callback,
            maxPathLength=1,
            maxHitsPerRay=1,
            nRNGSamples=nRNG,
        )
        # save params
        self._photons = wavelengthSource
        self._camera = camera
        self._source = source
        self._target = target
        self._materials = materials
        self.setParams(maxTime=maxTime)

        # compile code
        preamble = createPreamble(DISABLE_SELF_SHADOWING=target is None)
        headers = {
            "callback.glsl": callback.sourceCode,
            "camera.glsl": camera.sourceCode,
            "photon.glsl": wavelengthSource.sourceCode,
            "ray.glsl": ray.sourceCode,
            "response.glsl": response.sourceCode,
            "rng.glsl": rng.sourceCode,
            "source.glsl": source.sourceCode,
            "target.glsl": "" if target is None else target.sourceCode,
            "volume.glsl": volume.backwardSourceCode,
            **materials.header,
        }
        code = compileShader("tracer/volume/direct.main.glsl", preamble, headers)
        # create program
        self._program = hp.Program(code)
        materials.bindParams(self._program)

    @Tracer.batchSize.setter
    def batchSize(self, value) -> None:
        assert Tracer.batchSize.fset is not None  # to make linter happy
        Tracer.batchSize.fset(self, value)
        self.setParam("_batchSize", value)

    @property
    def camera(self) -> Camera:
        """Camera producing camera rays"""
        return self._camera

    @property
    def materials(self) -> MaterialStore:
        """MaterialStore containing the medium used by the tracer"""
        return self._materials

    @property
    def source(self) -> LightSource:
        """Source producing rays"""
        return self._source

    @property
    def target(self) -> Target | None:
        """Optional target used for self shadowing"""
        return self._target

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Sampler for wavelengths"""
        return self._photons

    def _collectStages(self) -> list[tuple[str, PipelineStage]]:
        stages = [
            ("rng", self.rng),
            ("photons", self.wavelengthSource),
            ("source", self.source),
            ("camera", self.camera),
            ("tracer", self),
            ("callback", self.callback),
            ("response", self.response),
        ]
        if self.target is not None:
            stages.insert(4, ("target", self.target))
        return stages

    def run(self, i: int) -> list[hp.Command]:
        for _, stage in self.collectStages():
            stage.bindParams(self._program, i)
        groups = -(self.capacity // -512)
        return [self._program.dispatch(groups)]
