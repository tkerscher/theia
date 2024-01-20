import hephaistos as hp
from hephaistos import Program
from hephaistos.glsl import buffer_reference, vec3
from hephaistos.pipeline import PipelineStage, SourceCodeMixin

from ctypes import Structure, c_float, c_int32, c_uint32, addressof, memset, sizeof
from numpy.ctypeslib import as_array

from theia.estimator import HitResponse
from theia.light import LightSource
from theia.random import RNG
from theia.scene import RectBBox, Scene, SphereBBox
from theia.util import ShaderLoader, compileShader

from enum import IntEnum
from numpy.typing import NDArray
from typing import Dict, List, Optional, Set, Tuple, Type


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


class VolumeTracer(PipelineStage):
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
    scatterCoefficient: float, default=0.01
        Scatter coefficient used for sampling ray lengths. Tuning this parameter
        affects the time distribution of the hits.
    traceBBox: RectBBox, default=(-1000,1000)^3
        Boundary box marking limits beyond tracing of an individual ray is
        stopped
    maxTime: float, default=1000.0
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
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
        target: SphereBBox = SphereBBox((0.0, 0.0, 0.0), 1.0),
        scatterCoefficient: float = 0.01,
        traceBBox: RectBBox = RectBBox((-1000.0,) * 3, (1000.0,) * 3),
        maxTime: float = 1000.0,
        blockSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        super().__init__({"TraceParams": self.TraceParams})
        # save params
        self._batchSize = batchSize
        self._source = source
        self._response = response
        self._rng = rng
        self._callback = callback
        self._nScattering = nScattering
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
            preamble += f"#define BATCH_SIZE {batchSize}\n"
            preamble += f"#define BLOCK_SIZE {blockSize}\n"
            preamble += f"#define DIM_OFFSET {source.nRNGSamples}\n"
            preamble += f"#define N_LAMBDA {source.nLambda}\n"
            preamble += f"#define N_SCATTER {nScattering}\n\n"
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
    def nScattering(self) -> int:
        """Number of simulated scattering events"""
        return self._nScattering

    @property
    def response(self) -> HitResponse:
        """Response function processing each simulated hit"""
        return self._response

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


class SceneShadowTracer(PipelineStage):
    """
    Path tracer simulating light traveling through media originating at a light
    source and ending at detectors. Traces rays against the geometries defined
    in scene to simulate accurate intersections and obstructions. Depending on
    the geometry's material, rays may reflect or transmit through them.

    As a speed-up method, cast shadow rays from scatter points to the detector
    additionally to the original light ray. Note that shadow rays must directly
    hit the detector to make a contribution. If e.g. the detector is behind some
    other geometry the ray needs to pass through, shadow rays will never reach
    it. In that case you should use the `SceneWalkTracer` instead.

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
    scene: Scene
        Scene in which the rays are traced
    nScattering: int, default=6
        Number of simulated scattering events
    targetIdx: int, default=0
        Id of the detector, the tracer should try to hit.
        Hits on other detectors are ignored to make estimates easier.
    scatterCoefficient: float, default=0.01
        Scatter coefficient used for sampling ray lengths. Tuning this parameter
        affects the time distribution of the hits.
    maxTime: float, default=1000.0
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    blockSize: int, default=128
        Number of threads in a single work group
    disableVolumeBorder: bool, default=False
        Disables GPU code handling volume borders, which may improve performance
        if there are none. Settings this to `True` while the scene contains
        volume border will produce wrong results.
    disableTransmission: bool, default=False
        Disables GPU code handling transmission, which may improve performance.
        Rays will default to always reflect where possible.
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

    name = "Scene Shadow Tracer"

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
        *,
        callback: TraceEventCallback = EmptyEventCallback(),
        scene: Scene,
        nScattering: int = 6,
        targetIdx: int = 0,
        scatterCoefficient: float = 0.01,
        maxTime: float = 1000.0,
        blockSize: int = 128,
        disableVolumeBorder: bool = False,
        disableTransmission: bool = False,
        code: Optional[bytes] = None,
    ) -> None:
        super().__init__({"TraceParams": self.TraceParams})
        # save params
        self._batchSize = batchSize
        self._source = source
        self._callback = callback
        self._response = response
        self._rng = rng
        self._scene = scene
        self._blockSize = blockSize
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
        # check if scene defines targets
        if scene.targets is None:
            raise ValueError("provided scene has no targets defined")

        # compile code if needed
        if code is None:
            preamble = ""
            if disableVolumeBorder:
                preamble += "#define DISABLE_VOLUME_BORDER 1\n"
            if disableTransmission:
                preamble += "#define DISABLE_TRANSMISSION 1\n"
            preamble += f"#define BATCH_SIZE {batchSize}\n"
            preamble += f"#define BLOCK_SIZE {blockSize}\n"
            preamble += f"#define DIM_OFFSET {source.nRNGSamples}\n"
            preamble += f"#define N_LAMBDA {source.nLambda}\n"
            preamble += f"#define N_SCATTER {nScattering}\n\n"
            headers = {
                "callback.glsl": callback.sourceCode,
                "response.glsl": response.sourceCode,
                "rng.glsl": rng.sourceCode,
                "source.glsl": source.sourceCode,
            }
            code = compileShader("tracer.scene.shadow.glsl", preamble, headers)
        self._code = code
        # create program
        self._program = hp.Program(self._code)
        # bind scene
        self._program.bindParams(
            Targets=scene.targets,
            Geometries=scene.geometries,
            tlas=scene.tlas,
        )

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
    def nScattering(self) -> int:
        """Number of simulated scattering events"""
        return self._nScattering

    @property
    def response(self) -> HitResponse:
        """Response function processing each simulated hit"""
        return self._response

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

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        self.callback.bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        return [self._program.dispatch(self._groups)]


class SceneWalkTracer(PipelineStage):
    """
    Path tracer simulating light traveling through media originating at a light
    source and ending at detectors. Traces rays against the geometries defined
    in scene to simulate accurate intersections and obstructions. Depending on
    the geometry's material, rays may reflect or transmit through them.

    As a speed-up method at each volume scatter event by chance a new direction
    is either sampled from the scattering phase function or from the general
    direction of the target detector, thus increasing the chances of actually
    hitting the latter.

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
    scene: Scene
        Scene in which the rays are traced
    nScattering: int, default=6
        Number of simulated scattering events
    targetIdx: int, default=0
        Id of the detector, the tracer should try to hit.
        Hits on other detectors are ignored to make estimates easier.
    scatterCoefficient: float, default=0.01
        Scatter coefficient used for sampling ray lengths. Tuning this parameter
        affects the time distribution of the hits.
    targetSampleProb: float, default=0.2
        Probability of sampling the target instead of the scattering phase
        function in scattering events for determining the scattered direction.
    maxTime: float, default=1000.0
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    blockSize: int, default=128
        Number of threads in a single work group
    disableVolumeBorder: bool, default=False
        Disables GPU code handling volume borders, which may improve performance
        if there are none. Settings this to `True` while the scene contains
        volume border will produce wrong results.
    disableTransmission: bool, default=False
        Disables GPU code handling transmission, which may improve performance.
        Rays will default to always reflect where possible.
    disableMIS: bool, default=False
        Disables importance sampling the target. Setting this to True, should be
        preferred to setting targetSampleProb to zero. Automatically set to
        True, if the scene has no targets.
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
    targetSampleProb: float, default=0.2
        Probability of sampling the target instead of the scattering phase
        function in scattering events for determining the scattered direction
    maxTime: float
        Max total time including delay from the source and travel time, after
        which a ray gets stopped
    """

    name = "Scene Walk Tracer"

    class TraceParams(Structure):
        _fields_ = [
            ("targetIdx", c_uint32),
            ("_sceneMedium", buffer_reference),
            ("targetSampleProb", c_float),
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
        *,
        callback: TraceEventCallback = EmptyEventCallback(),
        scene: Scene,
        nScattering: int = 6,
        targetIdx: int = 0,
        scatterCoefficient: float = 0.01,
        targetSampleProb: float = 0.2,
        maxTime: float = 1000.0,
        blockSize: int = 128,
        disableVolumeBorder: bool = False,
        disableTransmission: bool = False,
        disableMIS: bool = False,
        code: Optional[bytes] = None,
    ) -> None:
        super().__init__({"TraceParams": self.TraceParams})
        # save params
        self._batchSize = batchSize
        self._callback = callback
        self._source = source
        self._response = response
        self._rng = rng
        self._scene = scene
        self._blockSize = blockSize
        self._misDisabled = disableMIS
        self._transmissionDisabled = disableTransmission
        self._volumeBorderDisabled = disableVolumeBorder
        self.setParams(
            targetIdx=targetIdx,
            _sceneMedium=scene.medium,
            targetSampleProb=targetSampleProb,
            scatterCoefficient=scatterCoefficient,
            _lowerBBoxCorner=scene.bbox.lowerCorner,
            _upperBBoxCorner=scene.bbox.upperCorner,
            maxTime=maxTime,
            _maxDist=scene.bbox.diagonal,
        )
        # calculate group size
        self._groups = -(batchSize // -blockSize)
        # check if we have any targets
        if scene.targets is None:
            disableMIS = True

        # compile code if needed
        if code is None:
            preamble = ""
            if disableVolumeBorder:
                preamble += "#define DISABLE_VOLUME_BORDER 1\n"
            if disableTransmission:
                preamble += "#define DISABLE_TRANSMISSION 1\n"
            if disableMIS:
                preamble += "#define DISABLE_MIS 1\n"
            preamble += f"#define BATCH_SIZE {batchSize}\n"
            preamble += f"#define BLOCK_SIZE {blockSize}\n"
            preamble += f"#define DIM_OFFSET {source.nRNGSamples}\n"
            preamble += f"#define N_LAMBDA {source.nLambda}\n"
            preamble += f"#define N_SCATTER {nScattering}\n\n"
            headers = {
                "callback.glsl": callback.sourceCode,
                "response.glsl": response.sourceCode,
                "rng.glsl": rng.sourceCode,
                "source.glsl": source.sourceCode,
            }
            code = compileShader("tracer.scene.walker.glsl", preamble, headers)
        self._code = code
        # create program
        self._program = hp.Program(self._code)
        # bind scene
        self._program.bindParams(
            Geometries=scene.geometries,
            tlas=scene.tlas,
        )
        if not disableMIS:
            self._program.bindParams(Targets=scene.targets)

    @property
    def misDisabled(self) -> bool:
        """Wether MIS was disabled"""
        return self._misDisabled

    @property
    def transmissionDisabled(self) -> bool:
        """Wether transmission was disabled"""
        return self._transmissionDisabled

    @property
    def volumeBorderDisabled(self) -> bool:
        """Wether volume border support was disabled"""
        return self._volumeBorderDisabled

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
    def nScattering(self) -> int:
        """Number of simulated scattering events"""
        return self._nScattering

    @property
    def response(self) -> HitResponse:
        """Response function processing each simulated hit"""
        return self._response

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

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        self.callback.bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        return [self._program.dispatch(self._groups)]
