import hephaistos as hp
from hephaistos import Program
from hephaistos.glsl import buffer_reference, vec3
from hephaistos.pipeline import PipelineStage, SourceCodeMixin

from ctypes import Structure, c_float, c_int32, c_uint32, addressof, memset, sizeof
from numpy.ctypeslib import as_array

from theia.camera import CameraRaySource
from theia.estimator import HitResponse
from theia.light import LightSource
from theia.random import RNG
from theia.scene import RectBBox, Scene, SphereBBox
from theia.util import ShaderLoader, compileShader
import theia.units as u

import warnings

from enum import IntEnum
from numpy.typing import NDArray
from typing import Dict, List, Optional, Set, Tuple, Type


__all__ = [
    "BidirectionalPathTracer",
    "EmptyEventCallback",
    "EventResultCode",
    "EventStatisticCallback",
    "SceneTracer",
    "Tracer",
    "TraceEventCallback",
    "TrackRecordCallback",
    "VolumeTracer",
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
        params: Dict[str, Type[Structure]] = {},
        extra: Set[str] = set(),
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
    retrieve: bool, default=True
        Wether to retrieve the tracks from the device

    Note
    ----
    Mind that the total length is 1 + `nScatter` since it includes the
    start point.
    """

    name = "Track Recorder"

    def __init__(self, capacity: int, length: int, *, retrieve: bool = True) -> None:
        super().__init__()
        # save params
        self._capacity = capacity
        self._length = length
        self._retrieve = retrieve

        # allocate memory
        words = capacity * length * 4 + capacity * 2  # track + length + codes
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
    def retrieve(self) -> bool:
        """Wether to retrieve the tracks from the device"""
        return self._retrieve

    # sourceCode via descriptor
    _sourceCode = ShaderLoader("callback.track.glsl")

    @property
    def sourceCode(self) -> str:
        preamble = ""
        preamble += f"#define TRACK_COUNT {self.capacity}\n"
        preamble += f"#define TRACK_LENGTH {self.length}\n"
        return preamble + self._sourceCode

    @property
    def tensor(self) -> hp.Tensor:
        """Tensor containing the tracks"""
        return self._tensor

    def result(self, i: int) -> Tuple[NDArray, NDArray]:
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
        n = self.length * self.capacity * 4
        pTracks = (c_float * n).from_address(adr + 8 * self.capacity)
        # construct numpy array
        lengths = as_array(pLengths)
        codes = as_array(pCodes)
        tracks = as_array(pTracks).reshape((4, self.length, self.capacity))
        # change from (coords, ray, event) -> (ray, event, coords)
        tracks = tracks.transpose(2, 1, 0)
        return tracks, lengths, codes

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        program.bindParams(TrackBuffer=self._tensor)

    def run(self, i: int) -> List[hp.Command]:
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
        params: Dict[str, type[Structure]] = {},
        extra: Set[str] = set(),
        *,
        maxHits: int,
        normalization: float,
        nRNGSamples: int,
    ) -> None:
        super().__init__(params, extra)
        # save props
        self._response = response
        self._maxHits = maxHits
        self._normalization = normalization
        self._nRNGSamples = nRNGSamples
        # prepare response
        response.prepare(maxHits)

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
    def response(self) -> HitResponse:
        """Response function processing each simulated hit"""
        return self._response


class VolumeTracer(Tracer):
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
    response: HitResponse
        Response function processing each simulated hit
    rng: RNG
        Generator for creating random numbers
    callback: TraceEventCallback, default=EmptyEventCallback()
        Callback called for each tracing event. See `TraceEventCallback`
    medium: int, default=0
        device address of the medium the scene is emerged in, e.g. the address
        of a water medium for an underwater simulation.
        Defaults to zero specifying vacuum.
    nScattering: int, default=6
        Number of simulated scattering events
    target: SphereBBox, default=((0,0,0), r=1.0)
        Sphere the tracer targets
    scatterCoefficient: float, default=0.01 1/m
        Scatter coefficient used for sampling ray lengths. Tuning this parameter
        affects the time distribution of the hits.
    traceBBox: RectBBox, default=(-1km,1km)^3
        Boundary box marking limits beyond tracing of an individual ray is
        stopped
    maxTime: float, default=1000.0 ns
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    disableDirectLighting: bool, default=False
        Whether to ignore contributions from direct lighting, i.e. light paths
        with no scattering. If True, one can use a more performant direct
        lighting estimator in addition to increase overall performance.
    disableTargetSampling: bool, default=False
        Whether to disable sampling the target after scatter event. Disabling it
        means that targets will only be hit by chance and it is therefore
        recommended to not disable this as it hugely improves the performance.
        The simulated light path is not affected by this.
    blockSize: int, default=128
        Number of threads in a single work group
    code: Optional[bytes], default=None
        Cached compiled byte code. If `None` compiles it from source.
        The given byte code is not checked and must match the given
        configuration

    Stage Parameters
    ----------------
    targetPosition: vec3
        Center of the target sphere
    targetRadius: float
        Radius of the target sphere
    scatterCoefficient: float
        Scatter coefficient used for sampling ray lengths
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

    name = "Empty Scene Tracer"

    class TraceParams(Structure):
        _fields_ = [
            ("targetPosition", vec3),
            ("targetRadius", c_float),
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
        response: HitResponse,
        rng: RNG,
        *,
        callback: TraceEventCallback = EmptyEventCallback(),
        medium: int = 0,
        nScattering: int = 6,
        target: SphereBBox = SphereBBox((0.0, 0.0, 0.0), 1.0 * u.m),
        scatterCoefficient: float = 0.01 / u.m,
        traceBBox: RectBBox = RectBBox((-1.0 * u.km,) * 3, (1.0 * u.km,) * 3),
        maxTime: float = 1000.0 * u.ns,
        disableDirectLighting: bool = False,
        disableTargetSampling: bool = False,
        blockSize: int = 128,
        code: Optional[bytes] = None,
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
        # init tracer
        super().__init__(
            response,
            {"TraceParams": self.TraceParams},
            maxHits=maxHits,
            normalization=1.0 / batchSize,
            nRNGSamples=source.nRNGSamples + rngStride * pathLength,
        )
        # save params
        self._batchSize = batchSize
        self._source = source
        self._rng = rng
        self._callback = callback
        self._nScattering = nScattering
        self._directLightingDisabled = disableDirectLighting
        self._targetSamplingDisabled = disableTargetSampling
        self._blockSize = blockSize
        self.setParams(
            targetPosition=target.center,
            targetRadius=target.radius,
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
            # create preamble
            preamble = ""
            if disableDirectLighting:
                preamble += "#define DISABLE_DIRECT_LIGHTING 1\n"
            if disableTargetSampling:
                preamble += "#define DISABLE_MIS 1\n"
            preamble += f"#define BATCH_SIZE {batchSize}\n"
            preamble += f"#define BLOCK_SIZE {blockSize}\n"
            preamble += f"#define DIM_OFFSET {source.nRNGSamples}\n"
            preamble += f"#define PATH_LENGTH {pathLength}\n\n"
            headers = {
                "callback.glsl": callback.sourceCode,
                "response.glsl": response.sourceCode,
                "rng.glsl": rng.sourceCode,
                "source.glsl": source.sourceCode,
            }
            code = compileShader("tracer.volume.glsl", preamble, headers)
        self._code = code
        # create program
        self._program = hp.Program(self._code)

    @property
    def batchSize(self) -> int:
        """Number of rays to simulate per run"""
        return self._batchSize

    @property
    def blockSize(self) -> int:
        """Number of threads in a single work group"""
        return self._blockSize

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
    def target(self) -> SphereBBox:
        """Sphere the tracer targets"""
        return SphereBBox(
            self.getParam("targetPosition"),
            self.getParam("targetRadius"),
        )

    @target.setter
    def target(self, value: SphereBBox) -> None:
        self.setParams(targetPosition=value.center, targetRadius=value.radius)

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

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        self.callback.bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        return [self._program.dispatch(self._groups)]


class SceneTracer(Tracer):
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
    scatterCoefficient: float, default=0.01 1/m
        Scatter coefficient used for sampling ray lengths. Tuning this parameter
        affects the time distribution of the hits.
    maxTime: float, default=1000.0 ns
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    disableDirectLighting: bool, default=False
        Whether to ignore contributions from direct lighting, i.e. light paths
        with no scattering. If True, one can use a more performant direct
        lighting estimator in addition to increase overall performance.
    disableTargetSampling: bool, default=False
        Whether to disable sampling the target after scatter event. Disabling it
        means that targets will only be hit by chance and it is therefore
        recommended to not disable this as it hugely improves the performance.
        The simulated light path is not affected by this.
    disableTransmission: bool, default=False
        Disables GPU code handling transmission, which may improve performance.
        Rays will default to always reflect where possible.
    disableVolumeBorder: bool, default=False
        Disables GPU code handling volume borders, which may improve performance
        if there are none. Settings this to `True` while the scene contains
        volume border will produce wrong results.
    blockSize: int, default=128
        Number of threads in a single work group
    code: Optional[bytes], default=None
        Cached compiled byte code. If `None` compiles it from source.
        The given byte code is not checked and must match the given
        configuration

    Stage Parameters
    ----------------
    targetIdx: int
        Id of the detector, the tracer should try to hit.
    scatterCoefficient: float
        Scatter coefficient used for sampling ray lengths
    maxTime: float
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    """

    name = "Scene Tracer"

    class TraceParams(Structure):
        _fields_ = [
            ("targetIdx", c_uint32),
            ("_sceneMedium", buffer_reference),
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
        response: HitResponse,
        rng: RNG,
        scene: Scene,
        *,
        callback: TraceEventCallback = EmptyEventCallback(),
        maxPathLength: int = 6,
        targetIdx: int = 0,
        scatterCoefficient: float = 0.01 / u.m,
        maxTime: float = 1000.0 * u.ns,
        disableDirectLighting: bool = False,
        disableTargetSampling: bool = False,
        disableTransmission: bool = False,
        disableVolumeBorder: bool = False,
        blockSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        # check if ray tracing was enabled
        if not hp.isRaytracingEnabled():
            raise RuntimeError("Ray tracing is not supported on this system!")
        # disable MIS if there are no targets
        if scene.targets is None and not disableTargetSampling:
            warnings.warn(
                "Target sampling was requested but scene has no targets! "
                "Disabling target sampling which will most likely hurt performance."
            )
            disableTargetSampling = True

        # calculate max hits
        maxHits = batchSize * (maxPathLength - 1)
        if not disableTargetSampling:
            maxHits *= 2
        if not disableDirectLighting:
            maxHits += batchSize
        # init tracer
        rngStride = 4 if disableTargetSampling else 8
        super().__init__(
            response,
            {"TraceParams": self.TraceParams},
            maxHits=maxHits,
            normalization=1.0 / batchSize,
            nRNGSamples=source.nRNGSamples + rngStride * maxPathLength,
        )

        # save params
        self._batchSize = batchSize
        self._source = source
        self._callback = callback
        self._rng = rng
        self._scene = scene
        self._blockSize = blockSize
        self._maxPathLength = maxPathLength
        self._directLightingDisabled = disableDirectLighting
        self._transmissionDisabled = disableTransmission
        self._targetSamplingDisabled = disableTargetSampling
        self._volumeBorderDisabled = disableVolumeBorder
        self.setParams(
            targetIdx=targetIdx,
            scatterCoefficient=scatterCoefficient,
            _sceneMedium=scene.medium,
            maxTime=maxTime,
            _lowerBBoxCorner=scene.bbox.lowerCorner,
            _upperBBoxCorner=scene.bbox.upperCorner,
            _maxDist=scene.bbox.diagonal,
        )
        # calculate group size
        self._groups = -(batchSize // -blockSize)

        # compile code if needed
        if code is None:
            preamble = ""
            if disableDirectLighting:
                preamble += "#define DISABLE_DIRECT_LIGHTING 1\n"
            if disableTargetSampling:
                preamble += "#define DISABLE_MIS 1\n"
            if disableTransmission:
                preamble += "#define DISABLE_TRANSMISSION 1\n"
            if disableVolumeBorder:
                preamble += "#define DISABLE_VOLUME_BORDER 1\n"
            preamble += f"#define BATCH_SIZE {batchSize}\n"
            preamble += f"#define BLOCK_SIZE {blockSize}\n"
            preamble += f"#define DIM_OFFSET {source.nRNGSamples}\n"
            preamble += f"#define PATH_LENGTH {maxPathLength}\n\n"
            headers = {
                "callback.glsl": callback.sourceCode,
                "response.glsl": response.sourceCode,
                "rng.glsl": rng.sourceCode,
                "source.glsl": source.sourceCode,
            }
            code = compileShader("tracer.scene.glsl", preamble, headers)
        self._code = code
        # create program
        self._program = hp.Program(self._code)
        # bind scene
        self._program.bindParams(Geometries=scene.geometries, tlas=scene.tlas)
        if not disableTargetSampling:
            self._program.bindParams(Targets=scene.targets)

    @property
    def batchSize(self) -> int:
        """Number of rays to simulate per run"""
        return self._batchSize

    @property
    def blockSize(self) -> int:
        """Number of threads in a single work group"""
        return self._blockSize

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
    def targetSamplingDisabled(self) -> bool:
        """Whether target sampling is disabled"""
        return self._targetSamplingDisabled

    @property
    def transmissionDisabled(self) -> bool:
        """Whether transmission is disabled"""
        return self._transmissionDisabled

    @property
    def volumeBorderDisabled(self) -> bool:
        """Whether volume borders are disabled"""
        return self._volumeBorderDisabled

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        self.callback.bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
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
    camera: CameraRaySource
        Source producing camera rays
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
    lightPathLength: int, default=6
        Length of the light subpath
    cameraPathLength: int, default=6
        Length of the camera subpath
    targetIdx: int, default=0
        Id of the detector, the tracer should generate hits for.
        Hits on other detectors are ignored to make estimates easier.
    scatterCoefficient: float, default=0.01 1/m
        Scatter coefficient used for sampling ray lengths. Tuning this parameter
        affects the time distribution of the hits.
    maxTime: float, default=1000.0 ns
        Max total time including delay from each source and travel time, after
        which no responses are generated.
    cameraMedium: Optional[int], default=None
        Medium the camera is submerged in. If `None`, same as `scene.medium`.
    disableDirectLighting: bool, default=False
        Whether to ignore contributions from direct lighting, i.e. light paths
        with no scattering. If True, one can use a more performant direct
        lighting estimator in addition to increase overall performance.
    disableLightPathResponse: bool, default=False
        Whether the light subpath can generate hits on its own, i.e. before
        connecting with the camera subpath.
    disableTransmission: bool, default=False
        Disables GPU code handling transmission, which may improve performance.
        Rays will default to always reflect where possible.
    disableVolumeBorder: bool, default=False
        Disables GPU code handling volume borders, which may improve performance
        if there are none. Settings this to `True` while the scene contains
        volume border will produce wrong results.
    blockSize: int, default=128
        Number of threads in a single work group
    code: Optional[bytes], default=None
        Cached compiled byte code. If `None` compiles it from source.
        The given byte code is not checked and must match the given
        configuration

    Stage Parameters
    ----------------
    targetIdx: int
        Id of the detector, the tracer should generate hits for.
    scatterCoefficient: float
        Scatter coefficient used for sampling ray lengths
    maxTime: float
        Max total time including delay from each source and travel time, after
        which no responses are generated.
    """

    name = "Bidirectional Path Tracer"

    class TraceParams(Structure):
        _fields_ = [
            ("targetIdx", c_uint32),
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
        camera: CameraRaySource,
        response: HitResponse,
        rng: RNG,
        scene: Scene,
        *,
        callback: TraceEventCallback = EmptyEventCallback(),
        lightPathLength: int = 6,
        cameraPathLength: int = 6,
        targetIdx: int = 0,
        scatterCoefficient: float = 0.01 / u.m,
        maxTime: float = 1000.0 * u.ns,
        cameraMedium: Optional[int] = None,
        disableDirectLighting: bool = False,
        disableLightPathResponse: bool = False,
        disableTransmission: bool = False,
        disableVolumeBorder: bool = False,
        blockSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        # check if ray tracing was enabled
        if not hp.isRaytracingEnabled():
            raise RuntimeError("Ray tracing is not supported on this system")
        # calculate max hits
        maxHits = lightPathLength * cameraPathLength
        if not disableLightPathResponse:
            maxHits += lightPathLength - 1
        if not disableDirectLighting:
            maxHits += 1
        maxHits *= batchSize
        # calculate rng samples
        nRNG = source.nRNGSamples + camera.nRNGSamples
        nRNG += (lightPathLength + cameraPathLength) * 4
        # init tracer
        super().__init__(
            response,
            {"TraceParams": self.TraceParams},
            maxHits=maxHits,
            normalization=(1.0 / batchSize),
            nRNGSamples=nRNG,
        )

        # save params
        self._batchSize = batchSize
        self._source = source
        self._camera = camera
        self._callback = callback
        self._rng = rng
        self._scene = scene
        self._blockSize = blockSize
        self._lightPathLength = lightPathLength
        self._cameraPathLength = cameraPathLength
        self._directLightingDisabled = disableDirectLighting
        self._lightPathResponseDisabled = disableLightPathResponse
        self._transmissionDisabled = disableTransmission
        self._volumeBorderDisabled = disableVolumeBorder
        self.setParams(
            targetIdx=targetIdx,
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
            preamble = ""
            if disableDirectLighting:
                preamble += "#define DISABLE_DIRECT_LIGHTING 1\n"
            if disableLightPathResponse:
                preamble += "#define DISABLE_LIGHT_PATH_RESPONSE 1\n"
            if disableTransmission:
                preamble += "#define DISABLE_TRANSMISSION 1\n"
            if disableVolumeBorder:
                preamble += "#define DISABLE_VOLUME_BORDER 1\n"
            preamble += f"#define BATCH_SIZE {batchSize}\n"
            preamble += f"#define BLOCK_SIZE {blockSize}\n"
            preamble += f"#define DIM_OFFSET_LIGHT {source.nRNGSamples}\n"
            preamble += f"#define DIM_OFFSET_CAMERA {camera.nRNGSamples}\n"
            preamble += f"#define LIGHT_PATH_LENGTH {lightPathLength}\n"
            preamble += f"#define CAMERA_PATH_LENGTH {cameraPathLength}\n\n"
            headers = {
                "callback.glsl": callback.sourceCode,
                "response.glsl": response.sourceCode,
                "rng.glsl": rng.sourceCode,
                "source.glsl": source.sourceCode,
                "camera.glsl": camera.sourceCode,
            }
            code = compileShader("tracer.bidirectional.glsl", preamble, headers)
        self._code = code
        # create program
        self._program = hp.Program(self._code)
        # bind scene
        self._program.bindParams(Geometries=scene.geometries, tlas=scene.tlas)

    @property
    def batchSize(self) -> int:
        """Number of rays to simulate per run"""
        return self._batchSize

    @property
    def blockSize(self) -> int:
        """Number of threads in a single work group"""
        return self._blockSize

    @property
    def callback(self) -> TraceEventCallback:
        """Callback called for each tracing event"""
        return self._callback

    @property
    def camera(self) -> CameraRaySource:
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
    def directLightingDisabled(self) -> bool:
        """Whether direct lighting is disabled"""
        return self._directLightingDisabled

    @property
    def lightPathLength(self) -> int:
        """Length of the light subpath"""
        return self._lightPathLength

    @property
    def lightPathResponseDisabled(self) -> bool:
        """Whether contributions are generated from the light path"""
        return self._lightPathResponseDisabled

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

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        self.callback.bindParams(self._program, i)
        self.camera.bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        return [self._program.dispatch(self._groups)]
