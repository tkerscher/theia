from __future__ import annotations

import hephaistos as hp
from hephaistos import Program
from hephaistos.glsl import buffer_reference, vec3
from hephaistos.pipeline import PipelineStage, SourceCodeMixin

from abc import abstractmethod
from ctypes import Structure, c_float, c_int32, c_uint32, addressof, memset, sizeof
from numpy.ctypeslib import as_array

from theia.camera import Camera
from theia.response import HitResponse, TraceConfig
from theia.light import LightSource, WavelengthSource
from theia.random import RNG
from theia.scene import RectBBox, Scene
from theia.target import Target, TargetGuide
from theia.util import ShaderLoader, compileShader, createPreamble
import theia.units as u

from enum import IntEnum
from numpy.typing import NDArray
from typing import Literal


__all__ = [
    "BidirectionalPathTracer",
    "EmptyEventCallback",
    "EventResultCode",
    "EventStatisticCallback",
    "SceneBackwardTracer",
    "SceneForwardTracer",
    "Tracer",
    "TraceEventCallback",
    "TrackRecordCallback",
    "VolumeBackwardTracer",
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

    sourceCode = ShaderLoader("callback.empty.glsl")


class EventStatisticCallback(TraceEventCallback):
    """
    Callback recording statistics about the event types, i.e. increments a
    global counter for each event type. Current statistic are read directly from
    the device, but may have a bad impact on currently running tracing.
    """

    name = "Event Statistic"

    class Statistic(Structure):
        _fields_ = [
            ("created", c_uint32),
            ("scattered", c_uint32),
            ("hit", c_uint32),
            ("detected", c_uint32),
            ("volume", c_uint32),
            ("lost", c_uint32),
            ("decayed", c_uint32),
            ("absorbed", c_uint32),
            ("missed", c_uint32),
            ("error", c_uint32),
        ]

    def __init__(self) -> None:
        super().__init__()
        # allocate mapped tensor for statistics
        self._tensor = hp.ByteTensor(sizeof(self.Statistic), mapped=True)
        if not self._tensor.isMapped:
            raise RuntimeError("Could not create mapped tensor")
        self._stat = self.Statistic.from_address(self._tensor.memory)
        self.reset()

    # sourceCode via descriptor
    sourceCode = ShaderLoader("callback.stat.glsl")

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

    def reset(self) -> None:
        """Resets the statistic"""
        memset(addressof(self._stat), 0, sizeof(self.Statistic))

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        program.bindParams(Statistics=self._tensor)


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
    polarized: bool, default=False
        Whether to save polarization state. Save unpolarized state and zero
        vector as reference frame if ray contains no polarization state.
    retrieve: bool, default=True
        Whether to retrieve the tracks from the device
    """

    name = "Track Recorder"

    def __init__(
        self,
        capacity: int,
        length: int,
        *,
        polarized: bool = False,
        retrieve: bool = True,
    ) -> None:
        super().__init__()
        # save params
        self._capacity = capacity
        self._length = length
        self._retrieve = retrieve
        self._polarized = polarized

        # allocate memory
        self._cols = 11 if polarized else 4
        words = capacity * length * self._cols + capacity * 2  # track + length + codes
        self._tensor = hp.ByteTensor(words * 4)
        self._buffer = [hp.RawBuffer(words * 4) for _ in range(2)]

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

    # sourceCode via descriptor
    _sourceCode = ShaderLoader("callback.track.glsl")

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            TRACK_COUNT=self.capacity,
            TRACK_LENGTH=self.length,
            TRACK_POLARIZED=self.polarized,
        )
        return preamble + self._sourceCode

    @property
    def tensor(self) -> hp.Tensor:
        """Tensor containing the tracks"""
        return self._tensor

    def result(self, i: int) -> tuple[NDArray, NDArray]:
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
            Last recorded result code per track
        """
        # fetch each data structures
        adr = self._buffer[i].address
        pLengths = (c_uint32 * self.capacity).from_address(adr)
        pCodes = (c_int32 * self.capacity).from_address(adr + 4 * self.capacity)
        n = self.length * self.capacity * self._cols
        pTracks = (c_float * n).from_address(adr + 8 * self.capacity)
        # construct numpy array
        lengths = as_array(pLengths)
        codes = as_array(pCodes)
        tracks = as_array(pTracks).reshape((self._cols, self.length, self.capacity))
        # change from (coords, ray, event) -> (ray, event, coords)
        tracks = tracks.transpose(2, 1, 0)
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
    Base class for tracing algorithms taking in sampled light rays and producing
    hits.
    """

    name = "Tracer"

    def __init__(
        self,
        response: HitResponse,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
        *,
        batchSize: int,
        blockSize: int,
        maxHits: int,
        normalization: float,
        nRNGSamples: int,
        polarized: bool,
    ) -> None:
        super().__init__(params, extra)
        # save props
        self._batchSize = batchSize
        self._blockSize = blockSize
        self._response = response
        self._maxHits = maxHits
        self._normalization = normalization
        self._nRNGSamples = nRNGSamples
        self._polarized = polarized
        # prepare response
        c = TraceConfig(batchSize, blockSize, maxHits, normalization, polarized)
        response.prepare(c)

    @property
    def batchSize(self) -> int:
        """Number of rays to simulate per run"""
        return self._batchSize

    @property
    def blockSize(self) -> int:
        """Number of threads in a single work group"""
        return self._blockSize

    @property
    def maxHits(self) -> int:
        """Maximum amount of hits the tracer can produce per run"""
        return self._maxHits

    @property
    def normalization(self) -> float:
        """
        Normalization factor that must be applied to each sample to get a
        correct estimate.
        """
        return self._normalization

    @property
    def nRNGSamples(self) -> int:
        """Amount of random numbers drawn per ray"""
        return self._nRNGSamples

    @property
    def polarized(self) -> bool:
        """Whether polarization effects are simulated"""
        return self._polarized

    @property
    def response(self) -> HitResponse:
        """Response function processing each simulated hit"""
        return self._response

    @abstractmethod
    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all pipeline stages involved with this tracer in the
        correct order suitable for creating a pipeline.
        """
        pass


class VolumeForwardTracer(Tracer):
    """
    Path tracer simulating light traveling through media originating at a light
    source and ending at a detector. It does NOT check for any intersection
    with a scene and assumes the target detector to be spherical. Can be run on
    hardware without ray tracing support and may be faster.

    Parameters
    ----------
    batchSize: int
        Number of rays to simulate per run. Note that a single ray may generate
        up to nScattering hits.
    source: LightSource
        Source producing light rays
    target: Target
        Target model used to generate hits.
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from
    response: HitResponse
        Response function processing each simulated hit
    rng: RNG
        Generator for creating random numbers
    medium: int
        Device address of the medium the scene is emerged in, e.g. the address
        of a water medium for an underwater simulation.
    callback: TraceEventCallback, default=EmptyEventCallback()
        Callback called for each tracing event. See `TraceEventCallback`
    nScattering: int, default=6
        Number of simulated scattering events
    target: SphereBBox, default=((0,0,0), r=1.0)
        Sphere the tracer targets
    scatterCoefficient: float | None, default=None
        Scatter coefficient used for sampling ray lengths. Tuning this parameter
        affects the time distribution of the hits. If None, tracer will use the
        scattering length of the media the ray currently propagates.
    traceBBox: RectBBox, default=(-1km,1km)^3
        Boundary box marking limits beyond tracing of an individual ray is
        stopped
    maxTime: float, default=1000.0 ns
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    polarized: bool, default=False
        Whether to simulate polarization effects.
    disableDirectLighting: bool, default=False
        Whether to ignore contributions from direct lighting, i.e. light paths
        with no scattering. If True, one can use a more performant direct
        lighting estimator in addition to increase overall performance.
    disableTargetSampling: bool, default=false
        Whether to disable sampling the target after scatter event. Disabling it
        means that targets will only be hit by chance and it is therefore
        recommended to not disable this as it hugely improves the performance.
        The simulated light path is not affected by this.
    blockSize: int, default=128
        Number of threads in a single work group
    code: bytes | None, default=None
        Cached compiled byte code. If `None` compiles it from source.
        The given byte code is not checked and must match the given
        configuration

    Stage Parameters
    ----------------
    scatterCoefficient: float
        Scatter coefficient used for sampling ray lengths. Zero or negative
        values will cause the tracer to use the current scattering length
        instead.
    medium: int
        device address of the medium the scene is emerged in, e.g. the
        address of a water medium for an underwater simulation.
        Defaults to zero specifying vacuum.
    lowerBBoxCorner: (float, float, float)
        Lower limit of the x,y,z coordinates a ray must stay above to not get
        stopped
    upperBBoxCorner: (float, float, float)
        Upper limit of the x,y,z coordinates a ray must stay below to not get
        stopped
    maxTime: float
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    """

    name = "Volume Forward Tracer"

    class TraceParams(Structure):
        _fields_ = [
            ("medium", buffer_reference),
            ("scatterCoefficient", c_float),
            ("lowerBBoxCorner", vec3),
            ("upperBBoxCorner", vec3),
            ("maxTime", c_float),
            ("_maxDist", c_float),
        ]

    def __init__(
        self,
        batchSize: int,
        source: LightSource,
        target: Target,
        wavelengthSource: WavelengthSource,
        response: HitResponse,
        rng: RNG,
        *,
        medium: int,
        callback: TraceEventCallback = EmptyEventCallback(),
        nScattering: int = 6,
        scatterCoefficient: float | None = None,
        traceBBox: RectBBox = RectBBox((-1.0 * u.km,) * 3, (1.0 * u.km,) * 3),
        maxTime: float = 1000.0 * u.ns,
        polarized: bool = False,
        disableDirectLighting: bool = False,
        disableTargetSampling: bool = False,
        blockSize: int = 128,
        code: bytes | None = None,
    ) -> None:
        # calculate max hits
        maxHits = nScattering
        if not disableTargetSampling:
            maxHits *= 2
        if not disableDirectLighting:
            maxHits += 1
        maxHits *= batchSize
        # MIS also scatters -> one less trace command
        pathLength = nScattering if disableTargetSampling else nScattering - 1
        rngStride = 3 if disableTargetSampling else 7
        nRNG = source.nRNGForward + wavelengthSource.nRNGSamples
        nRNG += rngStride * pathLength
        # init tracer
        super().__init__(
            response,
            {"TraceParams": self.TraceParams},
            batchSize=batchSize,
            blockSize=blockSize,
            maxHits=maxHits,
            normalization=1.0 / batchSize,
            nRNGSamples=nRNG,
            polarized=polarized,
        )
        # save params
        if scatterCoefficient is None:
            scatterCoefficient = 0.0
        self._source = source
        self._wavelengthSource = wavelengthSource
        self._rng = rng
        self._target = target
        self._callback = callback
        self._nScattering = nScattering
        self._directLightingDisabled = disableDirectLighting
        self._targetSamplingDisabled = disableTargetSampling
        self.setParams(
            scatterCoefficient=scatterCoefficient,
            medium=medium,
            lowerBBoxCorner=traceBBox.lowerCorner,
            upperBBoxCorner=traceBBox.upperCorner,
            maxTime=maxTime,
            _maxDist=traceBBox.diagonal,
        )
        # calculate group size
        self._groups = -(batchSize // -blockSize)

        # compile code if needed
        if code is None:
            preamble = createPreamble(
                BATCH_SIZE=batchSize,
                BLOCK_SIZE=blockSize,
                DISABLE_DIRECT_LIGHTING=disableDirectLighting,
                DISABLE_MIS=disableTargetSampling,
                PATH_LENGTH=pathLength,
                POLARIZATION=polarized,
            )
            headers = {
                "callback.glsl": callback.sourceCode,
                "response.glsl": response.sourceCode,
                "rng.glsl": rng.sourceCode,
                "source.glsl": source.sourceCode,
                "photon.glsl": wavelengthSource.sourceCode,
                "target.glsl": target.sourceCode,
            }
            code = compileShader("tracer.volume.forward.glsl", preamble, headers)
        self._code = code
        # create program
        self._program = hp.Program(self._code)

    @property
    def callback(self) -> TraceEventCallback:
        """Callback called for each tracing event"""
        return self._callback

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching"""
        return self._code

    @property
    def directLightingDisabled(self) -> bool:
        """Wether direct lighting is disabled"""
        return self._directLightingDisabled

    @property
    def nScattering(self) -> int:
        """Number of simulated scattering events"""
        return self._nScattering

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    @property
    def source(self) -> LightSource:
        """Source producing light rays"""
        return self._source

    @property
    def target(self) -> Target:
        """Target model used to determine hits"""
        return self._target

    @property
    def traceBBox(self) -> RectBBox:
        """Boundary box of simulated rays"""
        return RectBBox(
            self.getParam("lowerBBoxCorner"), self.getParam("upperBBoxCorner")
        )

    @property
    def targetSamplingDisabled(self) -> bool:
        """Whether target sampling is disabled"""
        return self._targetSamplingDisabled

    @traceBBox.setter
    def traceBBox(self, value: RectBBox) -> None:
        self.setParams(
            lowerBBoxCorner=value.lowerCorner,
            upperBBoxCorner=value.upperCorner,
            _maxDist=value.diagonal,
        )

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source used to sample wavelengths"""
        return self._wavelengthSource

    def collectStages(self) -> list[PipelineStage]:
        return [
            self.rng,
            self.wavelengthSource,
            self.source,
            self.target,
            self,
            self.callback,
            self.response,
        ]

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        self.setParam("_maxDist", self.traceBBox.diagonal)

    def run(self, i: int) -> list[hp.Command]:
        self._bindParams(self._program, i)
        self.callback.bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.wavelengthSource.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        self.target.bindParams(self._program, i)
        return [self._program.dispatch(self._groups)]


class VolumeBackwardTracer(Tracer):
    """
    Path tracer sampling paths starting at the camera and creating complete ones
    by sampling the light independently at each scatter point. It does NOT check
    for any intersection with a scene but may include self shadowing from a
    spherical detector. Can be run on hardware without ray tracing support and
    may be faster.

    Parameters
    ----------
    batchSize: int
        Number of rays to simulate per run. Note that a single ray may generate
        up to nScattering hits plus one if direct lighting is enabled.
    source: LightSource
        Source producing light rays. Must support backward mode.
    camera: Camera
        Source producing camera rays. If direct lighting is enabled, must
        support direct mode.
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from
    response: HitResponse
        Response function simulating the detector
    rng: RNG
        Generator for creating random numbers
    medium: int
        dDevice address of the medium the scene is emerged in, e.g. the address
        of a water medium for an underwater simulation.
    callback: TraceEventCallback, default=EmptyEventCallback()
        Callback called for each tracing event. See `TraceEventCallback`
    nScattering: int, default=6
        Number of simulated scattering events
    target: Optional[Target], default=None
        Target model used to determine self shadowing. If None, this test is
        disabled.
    scatterCoefficient: float | None, default=None
        Scatter coefficient used for sampling ray lengths. Tuning this parameter
        affects the time distribution of the hits. If None, tracer will use the
        scattering length of the media the ray currently propagates.
    traceBBox: RectBBox, default=(-1km,1km)^3
        Boundary box marking limits beyond tracing of an individual ray is
        stopped
    maxTime: float, default=1000.0 ns
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    polarized: bool, default=False
        Whether to simulate polarization effects.
    disableDirectLighting: bool, default=False
        Whether to ignore contributions from direct lighting, i.e. light paths
        with no scattering.
    blockSize: int, default=128
        Number of threads in a single work group
    code: bytes | None, default=None
        Cached compiled byte code. If `None` compiles it from source.
        The given byte code is not checked and must match the given
        configuration

    Stage Parameters
    ----------------
    scatterCoefficient: float
        Scatter coefficient used for sampling ray lengths. Zero or negative
        values will cause the tracer to use the current scattering length
        instead.
    medium: int
        device address of the medium the scene is emerged in, e.g. the
        address of a water medium for an underwater simulation.
        Defaults to zero specifying vacuum.
    lowerBBoxCorner: (float, float, float)
        Lower limit of the x,y,z coordinates a ray must stay above to not get
        stopped
    upperBBoxCorner: (float, float, float)
        Upper limit of the x,y,z coordinates a ray must stay below to not get
        stopped
    maxTime: float
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    """

    name = "Volume Backward Tracer"

    class TraceParams(Structure):
        _fields_ = [
            ("medium", buffer_reference),
            ("scatterCoefficient", c_float),
            ("lowerBBoxCorner", vec3),
            ("upperBBoxCorner", vec3),
            ("maxTime", c_float),
            ("_maxDist", c_float),
        ]

    def __init__(
        self,
        batchSize: int,
        source: LightSource,
        camera: Camera,
        wavelengthSource: WavelengthSource,
        response: HitResponse,
        rng: RNG,
        *,
        medium: int,
        callback: TraceEventCallback = EmptyEventCallback(),
        nScattering: int = 6,
        target: Target | None = None,
        scatterCoefficient: float | None = None,
        traceBBox: RectBBox = RectBBox((-1.0 * u.km,) * 3, (1.0 * u.km,) * 3),
        maxTime: float = 1000.0 * u.ns,
        polarized: bool = False,
        disableDirectLighting: bool = False,
        blockSize: int = 128,
        code: bytes | None = None,
    ) -> None:
        # check if source and camera support this mode
        if not source.supportBackward:
            raise ValueError("Light source does not support backward mode!")
        if not disableDirectLighting and not camera.supportDirect:
            raise ValueError("Camera does not support direct mode!")

        # Calculate max hits
        maxHits = nScattering
        if not disableDirectLighting:
            maxHits += 1
        maxHits *= batchSize
        # calculate number of rng samples
        rngStride = 3 + source.nRNGBackward
        rngPre = wavelengthSource.nRNGSamples + camera.nRNGSamples
        if not disableDirectLighting:
            rngPre += wavelengthSource.nRNGSamples
            rngPre += camera.nRNGDirect
            rngPre += source.nRNGBackward
        nRNG = rngPre + rngStride * nScattering
        # init tracer
        super().__init__(
            response,
            {"TraceParams": self.TraceParams},
            batchSize=batchSize,
            blockSize=blockSize,
            maxHits=maxHits,
            normalization=1.0 / batchSize,
            nRNGSamples=nRNG,
            polarized=polarized,
        )
        # save params
        if scatterCoefficient is None:
            scatterCoefficient = 0.0
        self._source = source
        self._camera = camera
        self._wavelengthSource = wavelengthSource
        self._rng = rng
        self._target = target
        self._callback = callback
        self._nScattering = nScattering
        self._directLightingDisabled = disableDirectLighting
        self.setParams(
            scatterCoefficient=scatterCoefficient,
            medium=medium,
            lowerBBoxCorner=traceBBox.lowerCorner,
            upperBBoxCorner=traceBBox.upperCorner,
            maxTime=maxTime,
            _maxDist=traceBBox.diagonal,
        )
        # calculate group size
        self._groups = -(batchSize // -blockSize)

        # compile code if needed
        if code is None:
            preamble = createPreamble(
                BATCH_SIZE=batchSize,
                BLOCK_SIZE=blockSize,
                DISABLE_DIRECT_LIGHTING=disableDirectLighting,
                DISABLE_SELF_SHADOWING=target is None,
                PATH_LENGTH=nScattering,
                POLARIZATION=polarized,
            )
            headers = {
                "callback.glsl": callback.sourceCode,
                "camera.glsl": camera.sourceCode,
                "light.glsl": source.sourceCode,
                "response.glsl": response.sourceCode,
                "rng.glsl": rng.sourceCode,
                "source.glsl": wavelengthSource.sourceCode,
                "target.glsl": "" if target is None else target.sourceCode,
            }
            code = compileShader("tracer.volume.backward.glsl", preamble, headers)
        self._code = code
        # create program
        self._program = hp.Program(self._code)

    @property
    def callback(self) -> TraceEventCallback:
        """Callback called for each tracing event"""
        return self._callback

    @property
    def camera(self) -> Camera:
        """Source producing camera rays"""
        return self._camera

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching"""
        return self._code

    @property
    def directLightingDisabled(self) -> bool:
        """Wether direct lighting is disabled"""
        return self._directLightingDisabled

    @property
    def nScattering(self) -> int:
        """Number of simulated scattering events"""
        return self._nScattering

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    @property
    def source(self) -> LightSource:
        """Source producing light rays"""
        return self._source

    @property
    def target(self) -> Target | None:
        """Target used for self shadowing. If None, this test is disabled."""
        return self._target

    @property
    def traceBBox(self) -> RectBBox:
        """Boundary box of simulated rays"""
        return RectBBox(
            self.getParam("lowerBBoxCorner"), self.getParam("upperBBoxCorner")
        )

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source used to sample wavelengths"""
        return self._wavelengthSource

    def collectStages(self) -> list[PipelineStage]:
        stages = [self.rng, self.wavelengthSource, self.source, self.camera]
        if self.target is not None:
            stages.append(self.target)
        stages.extend([self, self.callback, self.response])
        return stages

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        self.setParam("_maxDist", self.traceBBox.diagonal)

    def run(self, i: int) -> list[hp.Command]:
        self._bindParams(self._program, i)
        self.callback.bindParams(self._program, i)
        self.camera.bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.wavelengthSource.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        if self.target is not None:
            self.target.bindParams(self._program, i)
        return [self._program.dispatch(self._groups)]


class SceneForwardTracer(Tracer):
    """
    Path tracer simulating light traveling through media originating at a light
    source and ending at detectors. Traces rays against the geometries defined
    by the provided scene to simulate accurate intersections and shadowing.
    Depending on the geometry's material, rays may reflect or transmit through
    them.

    Parameters
    ----------
    batchSize: int
        Number of rays to simulate per run. A single ray may and most likely
        will produce multiple responses.
    source: LightSource
        Source producing light rays
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from
    response: HitResponse
        Response function simulating the detector
    rng: RNG
        Generator for creating random numbers
    scene: Scene
        Scene in which the rays are traced
    callback: TraceEventCallback, default=EmptyEventCallback()
        Callback called for each tracing event. See `TraceEventCallback`
    maxPathLength: int, default=6
        Maximum number of events per simulated ray. An event includes volume
        scatter and scene intersection.
    targetIdx: int, default=0
        Id of the detector, the tracer should try to hit.
        Hits on other detectors are ignored to make estimates easier.
    targetGuide: TargetGuide | None, default=None
        Optional target proxy used to sample scatter directions towards it.
    scatterCoefficient: float | None, default=None
        Scatter coefficient used for sampling ray lengths. Tuning this parameter
        affects the time distribution of the hits. If None, tracer will use the
        scattering length of the media the ray currently propagates.
    sourceMedium: int | None, default=None
        Medium surrounding the light source. If None, uses the scene`s medium.
    maxTime: float, default=1000.0 ns
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    polarized: bool, default=False
        Whether to simulate polarization effects.
    disableDirectLighting: bool, default=False
        Whether to ignore contributions from direct lighting, i.e. light paths
        with no scattering. If True, one can use a more performant direct
        lighting estimator in addition to increase overall performance.
    disableTransmission: bool, default=False
        Disables GPU code handling transmission, which may improve performance.
        Rays will default to always reflect where possible.
    disableVolumeBorder: bool, default=False
        Disables GPU code handling volume borders, which may improve performance
        if there are none. Settings this to `True` while the scene contains
        volume border will produce wrong results.
    blockSize: int, default=128
        Number of threads in a single work group
    code: bytes | None, default=None
        Cached compiled byte code. If `None` compiles it from source.
        The given byte code is not checked and must match the given
        configuration

    Stage Parameters
    ----------------
    targetIdx: int
        Id of the detector, the tracer should try to hit.
    scatterCoefficient: float
        Scatter coefficient used for sampling ray lengths. Zero or negative
        values will cause the tracer to use the current scattering length
        instead.
    maxTime: float
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    """

    name = "Scene Forward Tracer"

    class TraceParams(Structure):
        _fields_ = [
            ("targetIdx", c_uint32),
            ("sourceMedium", buffer_reference),
            ("scatterCoefficient", c_float),
            ("_lowerBBoxCorner", vec3),
            ("_upperBBoxCorner", vec3),
            ("maxTime", c_float),
            ("_maxDist", c_float),
        ]

    def __init__(
        self,
        batchSize: int,
        source: LightSource,
        wavelengthSource: WavelengthSource,
        response: HitResponse,
        rng: RNG,
        scene: Scene,
        *,
        callback: TraceEventCallback = EmptyEventCallback(),
        maxPathLength: int = 6,
        targetIdx: int = 0,
        targetGuide: TargetGuide | None = None,
        scatterCoefficient: float | None = None,
        sourceMedium: int | None = None,
        maxTime: float = 1000.0 * u.ns,
        polarized: bool = False,
        disableDirectLighting: bool = False,
        disableTransmission: bool = False,
        disableVolumeBorder: bool = False,
        blockSize: int = 128,
        code: bytes | None = None,
    ) -> None:
        # check if ray tracing was enabled
        if not hp.isRaytracingEnabled():
            raise RuntimeError("Ray tracing is not supported on this system!")

        # calculate max hits
        maxHits = batchSize * (maxPathLength - 1)
        rngStride = 4
        if targetGuide is not None:
            maxHits *= 2
            rngStride += targetGuide.nRNGSamples
        if not disableDirectLighting:
            maxHits += batchSize
        # init tracer
        nRNG = source.nRNGForward + wavelengthSource.nRNGSamples
        nRNG += rngStride * maxPathLength
        super().__init__(
            response,
            {"TraceParams": self.TraceParams},
            batchSize=batchSize,
            blockSize=blockSize,
            maxHits=maxHits,
            normalization=1.0 / batchSize,
            nRNGSamples=nRNG,
            polarized=polarized,
        )

        # fetch scene's medium if none is specifed
        if sourceMedium is None:
            sourceMedium = scene.medium
        # save params
        if scatterCoefficient is None:
            scatterCoefficient = 0.0
        self._source = source
        self._wavelengthSource = wavelengthSource
        self._callback = callback
        self._rng = rng
        self._scene = scene
        self._targetGuide = targetGuide
        self._maxPathLength = maxPathLength
        self._directLightingDisabled = disableDirectLighting
        self._transmissionDisabled = disableTransmission
        self._volumeBorderDisabled = disableVolumeBorder
        self.setParams(
            targetIdx=targetIdx,
            scatterCoefficient=scatterCoefficient,
            sourceMedium=sourceMedium,
            maxTime=maxTime,
            _lowerBBoxCorner=scene.bbox.lowerCorner,
            _upperBBoxCorner=scene.bbox.upperCorner,
            _maxDist=scene.bbox.diagonal,
        )
        # calculate group size
        self._groups = -(batchSize // -blockSize)

        # compile code if needed
        if code is None:
            preamble = createPreamble(
                BATCH_SIZE=batchSize,
                BLOCK_SIZE=blockSize,
                DISABLE_DIRECT_LIGHTING=disableDirectLighting,
                DISABLE_MIS=targetGuide is None,
                DISABLE_TRANSMISSION=disableTransmission,
                DISABLE_VOLUME_BORDER=disableVolumeBorder,
                PATH_LENGTH=maxPathLength,
                POLARIZATION=polarized,
            )
            guideCode = "" if targetGuide is None else targetGuide.sourceCode
            headers = {
                "callback.glsl": callback.sourceCode,
                "response.glsl": response.sourceCode,
                "rng.glsl": rng.sourceCode,
                "source.glsl": source.sourceCode,
                "photon.glsl": wavelengthSource.sourceCode,
                "target_guide.glsl": guideCode,
            }
            code = compileShader("tracer.scene.forward.glsl", preamble, headers)
        self._code = code
        # create program
        self._program = hp.Program(self._code)
        # bind scene
        self._program.bindParams(Geometries=scene.geometries, tlas=scene.tlas)

    @property
    def callback(self) -> TraceEventCallback:
        """Callback called for each tracing event"""
        return self._callback

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching"""
        return self._code

    @property
    def directLightingDisabled(self) -> bool:
        """Whether direct lighting is disabled"""
        return self._directLightingDisabled

    @property
    def maxPathLength(self) -> int:
        """Maximum number of events per simulated ray"""
        return self._maxPathLength

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    @property
    def scene(self) -> Scene:
        """Scene in which the rays are traced"""
        return self._scene

    @property
    def source(self) -> LightSource:
        """Source producing light rays"""
        return self._source

    @property
    def targetGuide(self) -> TargetGuide | None:
        """Optional target guide used for sampling scatter directions"""
        return self._targetGuide

    @property
    def transmissionDisabled(self) -> bool:
        """Whether transmission is disabled"""
        return self._transmissionDisabled

    @property
    def volumeBorderDisabled(self) -> bool:
        """Whether volume borders are disabled"""
        return self._volumeBorderDisabled

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source used to sample wavelengths"""
        return self._wavelengthSource

    def collectStages(self) -> list[PipelineStage]:
        stages = [self.rng, self.wavelengthSource, self.source]
        if self.targetGuide is not None:
            stages.append(self.targetGuide)
        stages.extend([self, self.callback, self.response])
        return stages

    def run(self, i: int) -> list[hp.Command]:
        self._bindParams(self._program, i)
        self.callback.bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.wavelengthSource.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        if self._targetGuide is not None:
            self._targetGuide.bindParams(self._program, i)
        return [self._program.dispatch(self._groups)]


class SceneBackwardTracer(Tracer):
    """
    Path tracer sampling paths starting at the camera and creating complete ones
    by sampling the light independently at each scatter point. Traces rays
    against the geometries defined by the provided scene to simulate accurate
    intersections and shadowing. Depending on the geometry's material rays may
    reflect or transmit through them.

    Parameters
    ----------
    batchSize: int
        Number of rays to simulate per run. Note that a single ray may generate
        up to `maxPathLength` hits plus one if direct lighting is enabled.
    source: LightSource
        Source producing light rays. Must support backward mode.
    camera: Camera
        Source producing camera rays. If direct lighting is enabled must support
        direct mode.
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from
    response: HitResponse
        Response function simulating the detector
    rng: RNG
        Generator for creating random numbers
    scene: Scene
        Scene in which the rays are traced
    callback: TraceEventCallback, default=EmptyEventCallback()
        Callback called for each tracing event. See `TraceEventCallback`.
    medium: Optional[int], default=None
        Medium the camera is emerged in. Defaults to the scene's medium.
        Overrides scene's medium if present.
    maxPathLength: int, default=6
        Maximum number of events per simulated ray. An event includes volume
        scatter and scene intersections.
    scatterCoefficient: float | None, default=None
        Scatter coefficient used for sampling ray lengths. Tuning this parameter
        affects the time distribution of the hits. If None, tracer will use the
        scattering length of the media the ray currently propagates.
    maxTime: float, default=1000.0 ns
        Max total time including delay from source, camera and travel time after
        which a ray gets stopped
    polarized: bool, default=False
        Whether to simulate polarization effects.
    disableDirectLighting: bool, default=False
        Whether to ignore contributions from direct lighting, i.e. light paths
        with no scattering. Must be supported by camera to be enabled.
    disableTransmission: bool, default=False
        Disables GPU code handling transmission, which may improve performance.
        Rays will default to always reflect where possible.
    disableVolumeBorder: bool, default=False
        Disables GPU code handling volume borders, which may improve performance
        if there are none. Settings this to `True` while the scene contains
        volume border will produce wrong results.
    blockSize: int, default=128
        Number of threads in a single work group
    code: bytes | None, default=None
        Cached compiled byte code. If `None` compiles it from source.
        The given byte code is not checked and must match the given
        configuration

    Stage Parameters
    ----------------
    scatterCoefficient: float
        Scatter coefficient used for sampling ray lengths. Zero or negative
        values will cause the tracer to use the current scattering length
        instead.
    maxTime: float
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    """

    name = "Scene Backward Tracer"

    class TraceParams(Structure):
        _fields_ = [
            ("_medium", buffer_reference),
            ("scatterCoefficient", c_float),
            ("_lowerBBoxCorner", vec3),
            ("_upperBBoxCorner", vec3),
            ("maxTime", c_float),
            ("_maxDist", c_float),
        ]

    def __init__(
        self,
        batchSize: int,
        source: LightSource,
        camera: Camera,
        wavelengthSource: WavelengthSource,
        response: HitResponse,
        rng: RNG,
        scene: Scene,
        *,
        callback: TraceEventCallback = EmptyEventCallback(),
        medium: int | None = None,
        maxPathLength: int = 6,
        scatterCoefficient: float | None = None,
        maxTime: float = 1000.0 * u.ns,
        polarized: bool = False,
        disableDirectLighting: bool = False,
        disableTransmission: bool = False,
        disableVolumeBorder: bool = False,
        blockSize: int = 128,
        code: bytes | None = None,
    ) -> None:
        # check if source and camera support this mode
        if not source.supportBackward:
            raise ValueError("Light source does not support backward mode!")
        if not disableDirectLighting and not camera.supportDirect:
            raise ValueError("Camera does not support direct mode!")

        # Calculate max hits
        maxHits = maxPathLength
        if not disableDirectLighting:
            maxHits += 1
        maxHits *= batchSize
        # calculate number of rng samples
        rngStride = 3 + source.nRNGBackward
        rngPre = wavelengthSource.nRNGSamples + camera.nRNGSamples
        if not disableDirectLighting:
            rngPre += wavelengthSource.nRNGSamples
            rngPre += camera.nRNGDirect
            rngPre += source.nRNGBackward
        nRNG = rngPre + rngStride * maxPathLength
        # init tracer
        super().__init__(
            response,
            {"TraceParams": self.TraceParams},
            batchSize=batchSize,
            blockSize=blockSize,
            maxHits=maxHits,
            normalization=1.0 / batchSize,
            nRNGSamples=nRNG,
            polarized=polarized,
        )
        # save params
        if scatterCoefficient is None:
            scatterCoefficient = 0.0
        self._source = source
        self._camera = camera
        self._wavelengthSource = wavelengthSource
        self._rng = rng
        self._scene = scene
        self._callback = callback
        self._maxPathLength = maxPathLength
        self._directLightingDisabled = disableDirectLighting
        self._transmissionDisabled = disableTransmission
        self._volumeBorderDisabled = disableVolumeBorder
        self.setParams(
            scatterCoefficient=scatterCoefficient,
            _medium=scene.medium if medium is None else medium,
            maxTime=maxTime,
            _lowerBBoxCorner=scene.bbox.lowerCorner,
            _upperBBoxCorner=scene.bbox.upperCorner,
            _maxDist=scene.bbox.diagonal,
        )
        # calculate group size
        self._groups = -(batchSize // -blockSize)

        # compile code if needed
        if code is None:
            preamble = createPreamble(
                BATCH_SIZE=batchSize,
                BLOCK_SIZE=blockSize,
                DISABLE_DIRECT_LIGHTING=disableDirectLighting,
                DISABLE_TRANSMISSION=disableTransmission,
                DISABLE_VOLUME_BORDER=disableVolumeBorder,
                PATH_LENGTH=maxPathLength,
                POLARIZATION=polarized,
            )
            headers = {
                "callback.glsl": callback.sourceCode,
                "camera.glsl": camera.sourceCode,
                "light.glsl": source.sourceCode,
                "response.glsl": response.sourceCode,
                "rng.glsl": rng.sourceCode,
                "photon.glsl": wavelengthSource.sourceCode,
            }
            code = compileShader("tracer.scene.backward.glsl", preamble, headers)
        self._code = code
        # create program
        self._program = hp.Program(self._code)
        # bind scene
        self._program.bindParams(Geometries=scene.geometries, tlas=scene.tlas)

    @property
    def callback(self) -> TraceEventCallback:
        """Callback called for each tracing event"""
        return self._callback

    @property
    def camera(self) -> Camera:
        """Source producing camera rays"""
        return self._camera

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching"""
        return self._code

    @property
    def directLightingDisabled(self) -> bool:
        """Whether direct lighting is disabled"""
        return self._directLightingDisabled

    @property
    def maxPathLength(self) -> int:
        """Maximum number of events per simulated ray"""
        return self._maxPathLength

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    @property
    def scene(self) -> Scene:
        """Scene in which the rays are traced"""
        return self._scene

    @property
    def source(self) -> LightSource:
        """Source producing light rays"""
        return self._source

    @property
    def transmissionDisabled(self) -> bool:
        """Whether transmission is disabled"""
        return self._transmissionDisabled

    @property
    def volumeBorderDisabled(self) -> bool:
        """Whether volume borders are disabled"""
        return self._volumeBorderDisabled

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source used to sample wavelengths"""
        return self._wavelengthSource

    def collectStages(self):
        return [
            self.rng,
            self.wavelengthSource,
            self.source,
            self.camera,
            self,
            self.callback,
            self.response,
        ]

    def run(self, i: int) -> list[hp.Command]:
        self._bindParams(self._program, i)
        self.callback.bindParams(self._program, i)
        self.camera.bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.wavelengthSource.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        return [self._program.dispatch(self._groups)]


class DirectLightTracer(Tracer):
    """
    Path tracer directly connecting light source and camera without any
    scattering.

    Parameters
    ----------
    batchSize: int
        Number of connections to sample per run, each producing at most a single
        hit.
    source: LightSource
        Source producing light rays. Must support backward mode.
    camera: Camera
        Source producing camera rays. Must support direct mode.
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from
    response: HitResponse
        Response function simulating the detector
    rng: RNG
        Generator for creating random numbers
    scene: Optional[Scene], default=None
        Scene in which rays are trace to simulate shading effects.
    callback: TraceEventCallback, default=EmptyEventCallback()
        Callback called for each tracing event. See `TraceEventCallback`
    medium: Optional[int], default=None
        Medium the scene is emerged in. Defaults to the scene's medium.
        Must be provided if scene is `None`. Overrides scene's medium if
        present.
    maxTime: float, default=1000.0 ns
        Max total time including delay from each source and travel time, after
        which no responses are generated.
    polarized: bool, default=False
        Whether to simulate polarization effects.
    blockSize: int, default=128
        Number of threads in a single work group
    code: bytes | None, default=None
        Cached compiled byte code. If `None` compiles it from source.
        The given byte code is not checked and must match the given
        configuration

    Stage Parameters
    ----------------
    maxTime: float, default=1000.0 ns
        Max total time including delay from each source and travel time, after
        which no responses are generated.

    Note
    ----
    If no scene is given, there is still a check for light rays coming from the
    right direction via the detector normal ensuring a simplistic model of
    self-shadowing.
    """

    name = "Direct Light Tracer"

    class TraceParams(Structure):
        _fields_ = [
            ("medium", buffer_reference),
            ("_scatterCoefficient", c_float),
            ("_lowerBBoxCorner", vec3),
            ("_upperBBoxCorner", vec3),
            ("maxTime", c_float),
            ("_maxDist", c_float),
        ]

    def __init__(
        self,
        batchSize: int,
        source: LightSource,
        camera: Camera,
        wavelengthSource: WavelengthSource,
        response: HitResponse,
        rng: RNG,
        scene: Scene | None = None,
        *,
        callback: TraceEventCallback = EmptyEventCallback(),
        medium: int | None = None,
        maxTime: float = 1000.0 * u.ns,
        polarized: bool = False,
        blockSize: int = 128,
        code: bytes | None = None,
    ) -> None:
        # check if sources support this mode
        if not source.supportBackward:
            raise ValueError("Light source does not support backward mode")
        if not camera.supportDirect:
            raise ValueError("Camera does not support direct lighting")
        # check if there's a medium defined
        if scene is None and medium is None:
            raise ValueError("No medium was provided")
        # check for ray tracing if there's a scene
        if scene is not None and not hp.isRaytracingEnabled():
            raise RuntimeError("Ray tracing is not supported on this system")

        super().__init__(
            response,
            {"TraceParams": self.TraceParams},
            batchSize=batchSize,
            blockSize=blockSize,
            maxHits=batchSize,
            normalization=1.0 / batchSize,
            nRNGSamples=source.nRNGBackward + camera.nRNGDirect,
            polarized=polarized,
        )

        # save params
        self._source = source
        self._camera = camera
        self._wavelengthSource = wavelengthSource
        self._rng = rng
        self._scene = scene
        self._callback = callback
        # assemble scene
        bbox = RectBBox((float("-inf"),) * 3, (float("inf"),) * 3)
        maxDist = float("inf")
        if scene is not None:
            bbox = scene.bbox
            maxDist = scene.bbox.diagonal
            if medium is None:
                medium = scene.medium
        # set params
        self.setParams(
            medium=medium,
            _scatterCoefficient=float("NaN"),
            _lowerBBoxCorner=bbox.lowerCorner,
            _upperBBoxCorner=bbox.upperCorner,
            maxTime=maxTime,
            _maxDist=maxDist,
        )
        # calculate group size
        self._groups = -(batchSize // -blockSize)

        # compile code if needed
        if code is None:
            preamble = createPreamble(
                BATCH_SIZE=batchSize,
                BLOCK_SIZE=blockSize,
                DIM_PHOTON_OFFSET=wavelengthSource.nRNGSamples,
                DIM_CAM_OFFSET=wavelengthSource.nRNGSamples + camera.nRNGDirect,
                POLARIZATION=polarized,
                USE_SCENE=scene is not None,
            )
            headers = {
                "callback.glsl": callback.sourceCode,
                "camera.glsl": camera.sourceCode,
                "source.glsl": source.sourceCode,
                "photon.glsl": wavelengthSource.sourceCode,
                "response.glsl": response.sourceCode,
                "rng.glsl": rng.sourceCode,
            }
            code = compileShader("tracer.direct.glsl", preamble, headers)
        self._code = code
        # create program
        self._program = hp.Program(self._code)
        # bind scene if present
        if scene is not None:
            self._program.bindParams(tlas=scene.tlas)

    @property
    def callback(self) -> TraceEventCallback:
        """Callback called for each tracing event"""
        return self._callback

    @property
    def camera(self) -> Camera:
        """Source producing camera rays"""
        return self._camera

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching"""
        return self._code

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    @property
    def scene(self) -> Scene | None:
        """Scene in which the rays are traced"""
        return self._scene

    @property
    def source(self) -> LightSource:
        """Source producing light rays"""
        return self._source

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source used to sample wavelengths"""
        return self._wavelengthSource

    def collectStages(self) -> list[PipelineStage]:
        return [
            self.rng,
            self.wavelengthSource,
            self.source,
            self.camera,
            self,
            self.callback,
            self.response,
        ]

    def run(self, i: int) -> list[hp.Command]:
        self._bindParams(self._program, i)
        self.callback.bindParams(self._program, i)
        self.camera.bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        self.wavelengthSource.bindParams(self._program, i)
        return [self._program.dispatch(self._groups)]


class BidirectionalPathTracer(Tracer):
    """
    Path tracer simulating subpath from both the camera and light source
    creating complete light paths by establishing all possible connections
    between the two subpath. Suffers from higher variance but may show a much
    higher performance in difficult scenes like the detector being placed
    behind a refracting surface.

    Parameters
    ----------
    batchSize: int
        Number of subpath pairs to simulate per run, each producing multiple
        responses.
    source: LightSource
        Source producing light rays
    camera: Camera
        Source producing camera rays
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from
    response: HitResponse
        Response function simulating the detector
    rng: RNG
        Generator for creating random numbers
    scene: Scene
        Scene in which the rays are traced
    callback: TraceEventCallback, default=EmptyEventCallback()
        Callback called for each tracing event. See `TraceEventCallback`.
        Starts with the light sub path followed by the camera sub path.
        Each subpath starts with an `RAY_CREATED` event.
    callbackScope: "light"|"camera"|"both", default="both"
        Optionally limits the scope of callback calls to the specified subpath.
    lightPathLength: int, default=6
        Length of the light subpath
    cameraPathLength: int, default=6
        Length of the camera subpath
    targetIdx: int, default=0
        Id of the detector, the tracer should generate hits for.
        Hits on other detectors are ignored to make estimates easier.
    scatterCoefficient: float | None, default=None
        Scatter coefficient used for sampling ray lengths. Tuning this parameter
        affects the time distribution of the hits. If None, tracer will use the
        scattering length of the media the ray currently propagates.
    maxTime: float, default=1000.0 ns
        Max total time including delay from each source and travel time, after
        which no responses are generated.
    polarized: bool, default=False
        Whether to simulate polarization effects.
    cameraMedium: Optional[int], default=None
        Medium the camera is submerged in. If `None`, same as `scene.medium`.
    disableTransmission: bool, default=False
        Disables GPU code handling transmission, which may improve performance.
        Rays will default to always reflect where possible.
    disableVolumeBorder: bool, default=False
        Disables GPU code handling volume borders, which may improve performance
        if there are none. Settings this to `True` while the scene contains
        volume border will produce wrong results.
    blockSize: int, default=128
        Number of threads in a single work group
    code: bytes | None, default=None
        Cached compiled byte code. If `None` compiles it from source.
        The given byte code is not checked and must match the given
        configuration

    Stage Parameters
    ----------------
    scatterCoefficient: float
        Scatter coefficient used for sampling ray lengths. Zero or negative
        values will cause the tracer to use the current scattering length
        instead.
    maxTime: float
        Max total time including delay from each source and travel time, after
        which no responses are generated.

    Note
    ----
    The BidirectionalPathTracer samples only paths of at least length 2, i.e.
    only paths with at least two scattering events and thus misses energy from
    direct contribution and single scattering events.
    """

    name = "Bidirectional Path Tracer"

    class TraceParams(Structure):
        _fields_ = [
            ("_sceneMedium", buffer_reference),
            ("_cameraMedium", buffer_reference),
            ("scatterCoefficient", c_float),
            ("_lowerBBoxCorner", vec3),
            ("_upperBBoxCorner", vec3),
            ("maxTime", c_float),
            ("_maxDist", c_float),
        ]

    def __init__(
        self,
        batchSize: int,
        source: LightSource,
        camera: Camera,
        wavelengthSource: WavelengthSource,
        response: HitResponse,
        rng: RNG,
        scene: Scene,
        *,
        callback: TraceEventCallback = EmptyEventCallback(),
        callbackScope: Literal["light", "camera", "both"] = "both",
        lightPathLength: int = 6,
        cameraPathLength: int = 6,
        scatterCoefficient: float | None = None,
        maxTime: float = 1000.0 * u.ns,
        polarized: bool = False,
        cameraMedium: int | None = None,
        disableTransmission: bool = False,
        disableVolumeBorder: bool = False,
        blockSize: int = 128,
        code: bytes | None = None,
    ) -> None:
        # check if ray tracing was enabled
        if not hp.isRaytracingEnabled():
            raise RuntimeError("Ray tracing is not supported on this system")
        # calculate max hits
        maxHits = lightPathLength * cameraPathLength
        maxHits *= batchSize
        # calculate rng samples
        nRNG = source.nRNGForward + camera.nRNGSamples + wavelengthSource.nRNGSamples
        nRNG += (lightPathLength + cameraPathLength) * 4
        # init tracer
        super().__init__(
            response,
            {"TraceParams": self.TraceParams},
            batchSize=batchSize,
            blockSize=blockSize,
            maxHits=maxHits,
            normalization=(1.0 / batchSize),
            nRNGSamples=nRNG,
            polarized=polarized,
        )

        # save params
        if scatterCoefficient is None:
            scatterCoefficient = 0.0
        self._wavelengthSource = wavelengthSource
        self._source = source
        self._camera = camera
        self._callback = callback
        self._rng = rng
        self._scene = scene
        self._lightPathLength = lightPathLength
        self._cameraPathLength = cameraPathLength
        self._callbackScope = callbackScope
        self._transmissionDisabled = disableTransmission
        self._volumeBorderDisabled = disableVolumeBorder
        self.setParams(
            _sceneMedium=scene.medium,
            _cameraMedium=(scene.medium if cameraMedium is None else cameraMedium),
            scatterCoefficient=scatterCoefficient,
            maxTime=maxTime,
            _lowerBBoxCorner=scene.bbox.lowerCorner,
            _upperBBoxCorner=scene.bbox.upperCorner,
            _maxDist=scene.bbox.diagonal,
        )
        # calculate group size
        self._groups = -(batchSize // -blockSize)

        # compile code if needed
        if code is None:
            preamble = createPreamble(
                BATCH_SIZE=batchSize,
                BLOCK_SIZE=blockSize,
                CAMERA_PATH_LENGTH=cameraPathLength,
                DISABLE_CAMERA_CALLBACK=callbackScope == "light",
                DISABLE_LIGHT_CALLBACK=callbackScope == "camera",
                DISABLE_TRANSMISSION=disableTransmission,
                DISABLE_VOLUME_BORDER=disableVolumeBorder,
                LIGHT_PATH_LENGTH=lightPathLength,
                POLARIZATION=polarized,
            )
            headers = {
                "callback.glsl": callback.sourceCode,
                "response.glsl": response.sourceCode,
                "rng.glsl": rng.sourceCode,
                "light.glsl": source.sourceCode,
                "camera.glsl": camera.sourceCode,
                "photon.glsl": wavelengthSource.sourceCode,
            }
            code = compileShader("tracer.bidirectional.glsl", preamble, headers)
        self._code = code
        # create program
        self._program = hp.Program(self._code)
        # bind scene
        self._program.bindParams(Geometries=scene.geometries, tlas=scene.tlas)

    @property
    def callback(self) -> TraceEventCallback:
        """Callback called for each tracing event"""
        return self._callback

    @property
    def callbackScope(self) -> Literal["light", "camera", "both"]:
        """Scope the callback is limited to"""
        return self._callbackScope

    @property
    def camera(self) -> Camera:
        """Source generating camera rays"""
        return self._camera

    @property
    def cameraPathLength(self) -> int:
        """Length of the camera subpath"""
        return self._cameraPathLength

    @property
    def code(self) -> bytes:
        """Compiled source code. Can be used for caching"""
        return self._code

    @property
    def lightPathLength(self) -> int:
        """Length of the light subpath"""
        return self._lightPathLength

    @property
    def rng(self) -> RNG:
        """Generator for creating random numbers"""
        return self._rng

    @property
    def scene(self) -> Scene:
        """Scene in which the rays are traced"""
        return self._scene

    @property
    def source(self) -> LightSource:
        """Source producing light rays"""
        return self._source

    @property
    def transmissionDisabled(self) -> bool:
        """Whether transmission is disabled"""
        return self._transmissionDisabled

    @property
    def volumeBorderDisabled(self) -> bool:
        """Whether volume borders are disabled"""
        return self._volumeBorderDisabled

    @property
    def wavelengthSource(self) -> WavelengthSource:
        """Source used to sample wavelengths"""
        return self._wavelengthSource

    def collectStages(self):
        return [
            self.rng,
            self.wavelengthSource,
            self.source,
            self.camera,
            self,
            self.callback,
            self.response,
        ]

    def run(self, i: int) -> list[hp.Command]:
        self._bindParams(self._program, i)
        self.callback.bindParams(self._program, i)
        self.camera.bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        self.wavelengthSource.bindParams(self._program, i)
        return [self._program.dispatch(self._groups)]
