import hephaistos as hp
from hephaistos.glsl import buffer_reference, vec3
from hephaistos.pipeline import PipelineStage

from ctypes import Structure, c_float, c_uint32

from theia.estimator import HitResponse
from theia.light import LightSource
from theia.random import RNG
from theia.scene import RectBBox, Scene, SphereBBox
from theia.util import compileShader

from typing import List, Optional


class EmptySceneTracer(PipelineStage):
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
            ("scatterCoefficient", c_float),
            ("medium", buffer_reference),
            ("lowerBBoxCorner", vec3),
            ("upperBBoxCorner", vec3),
            ("maxTime", c_float),
        ]

    def __init__(
        self,
        batchSize: int,
        source: LightSource,
        response: HitResponse,
        rng: RNG,
        *,
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
                "rng.glsl": rng.sourceCode,
                "source.glsl": source.sourceCode,
                "response.glsl": response.sourceCode,
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
            lowerBBoxCorner=value.lowerCorner, upperBBoxCorner=value.upperCorner
        )

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._program, i)
        self.source.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        return [self._program.dispatch(self._groups)]


class SceneTracer(PipelineStage):
    """
    Path tracer simulating light traveling through media originating at a light
    source and ending at detectors. Traces rays against the geometries defined
    in scene to simulate accurate intersections and obstructions.

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
            ("scatterCoefficient", c_float),
            ("_sceneMedium", buffer_reference),
            ("maxTime", c_float),
            ("_lowerBBoxCorner", vec3),
            ("_upperBBoxCorner", vec3),
        ]

    def __init__(
        self,
        batchSize: int,
        source: LightSource,
        response: HitResponse,
        rng: RNG,
        *,
        scene: Scene,
        nScattering: int = 6,
        targetIdx: int = 0,
        scatterCoefficient: float = 0.01,
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
        self._scene = scene
        self._blockSize = blockSize
        self.setParams(
            targetIdx=targetIdx,
            scatterCoefficient=scatterCoefficient,
            _sceneMedium=scene.medium,
            maxTime=maxTime,
            _lowerBBoxCorner=scene.bbox.lowerCorner,
            _upperBBoxCorner=scene.bbox.upperCorner,
        )
        # calculate group size
        self._groups = -(batchSize // -blockSize)

        # compile code if needed
        if code is None:
            preamble = ""
            preamble += f"#define BATCH_SIZE {batchSize}\n"
            preamble += f"#define BLOCK_SIZE {blockSize}\n"
            preamble += f"#define DIM_OFFSET {source.nRNGSamples}\n"
            preamble += f"#define N_LAMBDA {source.nLambda}\n"
            preamble += f"#define N_SCATTER {nScattering}\n\n"
            headers = {
                "rng.glsl": rng.sourceCode,
                "source.glsl": source.sourceCode,
                "response.glsl": response.sourceCode,
            }
            code = compileShader("tracer.scene.glsl", preamble, headers)
        self._code = code
        # create program
        self._program = hp.Program(self._code)
        # bind tlas
        self._program.bindParams(
            Detectors=scene.detectors,
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
        self.source.bindParams(self._program, i)
        self.response.bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        return [self._program.dispatch(self._groups)]
