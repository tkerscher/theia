import hephaistos as hp

from hephaistos.glsl import vec3
from hephaistos.pipeline import PipelineStage
from hephaistos.queue import QueueTensor, clearQueue

import theia.items
from theia.scene import RectBBox, Scene, SphereBBox
from theia.random import RNG
from theia.util import compileShader

from ctypes import Structure, c_float, c_int32, c_uint32
from typing import Dict, List, Optional


class EmptySceneTracer(PipelineStage):
    """
    Path tracer simulating light traveling through media originating at a light
    source and ending at a detector. It does NOT check for any intersection
    with a scene and assumes the target detector to be spherical. Can be run on
    hardware without ray tracing support and may be faster.

    Parameters
    ----------
    capacity: int
        Maximum number of samples the tracer can process in a single run.
    rng: RNG
        Generator for creating random numbers.
    nScattering: int, default=6
        Number of simulated scattering events per ray and iteration.
    nIterations: int, default=1
        Number of iterations in the tracer. The tracer gets the chance to merge
        diverging execution between iterations, which might improve performance.
    target: SphereBBox, default=((0,0,0), r=1.0)
        Sphere the tracer should target the rays to hit.
    nPhotons: int, default=4
        Number of photons (i.e. wavelengths) in a single ray
    scatterCoefficient: float, default=10.0
        Scattering coefficient used for sampling. Can be used to tune the time
        distribution of ray hits
    traceBBox: RectBBox, default=(-1000,1000)^3
        Boundary bbox marking limits beyond tracing of an individual ray is
        stopped.
    maxTime: float, default=1000.0
        Max time after which traversing a ray stops
    clearHitQueue: bool, default=True
        Wether to clear the hit queue before each batch.
    keepRays: bool, default=False,
        Wether to keep the last state of simulated rays that have not hit the
        target yet.
    blockSize: int, default=128
        Number of threads in a single local work group (block size in CUDA)
    code: Optional[bytes], default=None
        cached compiled code. Must match the configuration and might be
        different depending on the local machine.
        If None, the code get's compiled from source.
    """

    name = "Empty Scene Tracer"

    class TraceParams(Structure):
        """Structure matching the shader's trace params"""

        _fields_ = [
            ("targetPosition", vec3),
            ("targetRadius", c_float),
            ("scatterCoefficient", c_float),
            ("lowerBBoxCorner", vec3),
            ("upperBBoxCorner", vec3),
            ("maxTime", c_float),
        ]

    class Push(Structure):
        """Structure matching the shader's push constants"""

        _fields_ = [("saveRays", c_int32)]

    def __init__(
        self,
        capacity: int,
        *,
        nScattering: int = 6,
        nIterations: int = 1,
        rng: RNG,
        target: SphereBBox = SphereBBox((0.0,) * 3, 1.0),
        nPhotons: int = 4,
        scatterCoefficient: float = 10.0,
        traceBBox: RectBBox = RectBBox((-1000.0,) * 3, (1000.0,) * 3),
        maxTime: float = 1000.0,
        clearHitQueue: bool = True,
        keepRays: bool = False,
        blockSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        super().__init__({"TraceParams": self.TraceParams})
        # save params
        self._rng = rng
        self._capacity = capacity
        self._nScattering = nScattering
        self._nIterations = nIterations
        self._nPhotons = nPhotons
        self.setParams(
            targetPosition=target.center,
            targetRadius=target.radius,
            scatterCoefficient=scatterCoefficient,
            maxTime=maxTime,
            lowerBBoxCorner=traceBBox.lowerCorner,
            upperBBoxCorner=traceBBox.upperCorner,
        )

        # calculate sizes
        self._groupSize = -(capacity // -blockSize)
        self._maxSamples = capacity * nScattering * nIterations
        # ray queue buffer flag
        self._clearHitQueue = clearHitQueue
        self._keepRays = keepRays
        bufferRayQueue = keepRays or nIterations > 1

        # compile code if needed
        if code is None:
            # create preamble
            preamble = ""
            preamble += f"#define BATCH_SIZE {blockSize}\n"
            preamble += f"#define HIT_QUEUE_SIZE {self._maxSamples}\n"
            preamble += f"#define N_PHOTONS {nPhotons}\n"
            preamble += f"#define N_SCATTER {nScattering}\n"
            preamble += f"#define QUEUE_SIZE {capacity}\n"
            if bufferRayQueue:
                preamble += f"#define OUTPUT_RAYS\n"
            # compile source code
            headers = {"rng.glsl": rng.sourceCode}
            code = compileShader("tracer.volume.glsl", preamble, headers)
        # save code
        self._code = code
        # create program
        self._program = hp.Program(code)

        # create queue items
        rayItem = theia.items.createRayQueueItem(nPhotons)
        hitItem = theia.items.createHitQueueItem(nPhotons)
        # create queues
        if bufferRayQueue:
            self._rayQueue = [QueueTensor(rayItem, capacity) for _ in range(2)]
        else:
            self._rayQueue = [QueueTensor(rayItem, capacity)]
        self._hitQueue = QueueTensor(hitItem, self._maxSamples)
        self._program.bindParams(HitQueueBuffer=self._hitQueue)

    @property
    def capacity(self) -> int:
        """number of light samples processed per run"""
        return self._capacity

    @property
    def code(self) -> Dict[str, bytes]:
        """
        Dictionary containing the compiled source codes used in the pipeline.
        Can be used to cache code allowing to skip the compilation.
        The configuration must match.
        """
        return self._code

    @property
    def nIterations(self) -> int:
        """
        Number of iterations in the tracer. The tracer gets the chance to merge
        diverging execution between iterations, which might improve performance.
        """
        return self._nIterations

    @property
    def nScattering(self) -> int:
        """max number of scattering events per ray"""
        return self._nScattering

    @property
    def nPhotons(self) -> int:
        """number of photons (wavelengths) per light ray/sample"""
        return self._nPhotons

    @property
    def maxSamples(self) -> int:
        """Maximum number of samples produced per batch"""
        return self.capacity * self.nScattering * self.nIterations

    @property
    def rng(self) -> RNG:
        """Generator used for sampling random numbers"""
        return self._rng

    @property
    def rayQueueIn(self) -> hp.Tensor:
        """Tensor containing the ray queue the tracing starts with"""
        return self._rayQueue[0]

    @property
    def rayQueueOut(self) -> Optional[hp.Tensor]:
        """
        Tensor containing the final rays the tracer stopped tracing,
        i.e. the initial rays minus the ones that directly hit the target.
        None, if the tracer does not save the
        """
        if self._keepRays:
            return self._rayQueue[(self._nIterations + 1) % 2]
        else:
            return None

    @property
    def keepRays(self) -> bool:
        """
        True, if the tracer saves the final rays the tracer stopped tracing,
        i.e. the initial rays minus the ones that directly hit the target.
        """
        return self._keepRays

    @property
    def clearHitQueue(self) -> bool:
        """Wether to clear the hit queue before each batch."""
        return self._clearHitQueue

    @property
    def hitQueue(self) -> hp.Tensor:
        """Tensor containing the hit queue"""
        return self._hitQueue

    # pipeline stage api

    def run(self, i: int) -> List[hp.Command]:
        cmd = []
        self._bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        # clear queue if needed
        if self.clearHitQueue:
            cmd.append(clearQueue(self.hitQueue))
        # check if we need the double buffering
        if not self.keepRays and self.nIterations == 1:
            self._program.bindParams(RayQueueBuffer=self._rayQueue[0])
            cmd.extend(
                [
                    self._program.dispatch(self._groupSize),
                    hp.flushMemory(),
                ]
            )
        else:
            # swap buffers as needed
            for it in range(self.nIterations):
                self._program.bindParams(
                    RayQueueBuffer=self._rayQueue[it % 2],
                    OutRayQueueBuffer=self._rayQueue[(it + 1) % 2],
                )
                save = self.keepRays or it < self.nIterations - 1
                push = self.Push(saveRays=(1 if save else 0))
                cmd.extend(
                    [
                        clearQueue(self._rayQueue[(it + 1) % 2]),
                        self._program.dispatchPush(bytes(push), self._groupSize),
                        hp.flushMemory(),
                    ]
                )
        # done
        return cmd


class SceneTracer(PipelineStage):
    """
    Path tracer simulating light traveling through media originating at a
    light source and ending at detectors. Traces rays against the geometries
    defined in scene to simulate accurate intersections and obstructions.

    Parameters
    ----------
    capacity: int
        Maximum number of samples the tracer can process in a single run.
    scene: Scene
        Scene to trace rays against
    rng: RNG
        Generator for sampling random numbers
    nScattering: int, default=6
        Number of scatter events (both surface and volume) to simulate per ray
    nIterations: int, default=1
        Number of iterations in the tracer. The tracer gets the chance to merge
        diverging execution between iterations, which might improve performance.
    nPhotons: int, default=4
        Number of photons (i.e. wavelengths) in single ray
    maxTime: float, default=1000.0
        Max time after which rays will be canceled
    scatteringCoefficient: float, default=10.0
        Scattering coefficient used for sampling. Can be used to tune the time
        distribution of ray hits
    targetIdx: int, default=0
        Id of the detector, the tracer should try to hit.
        Hits on other detectors are ignored to make the estimates simpler.
    clearHitQueue: bool, default=True
        Wether to clear the hit queue before each batch.
    keepRays: bool, default=False,
        Wether to keep the last state of simulated rays that have not hit the
        target yet.
    blockSize: int, default=128
        Number of threads in a single local work group (block size in CUDA)
    code: { str: bytes }, default={}
        Dictionary containing the compiled shader code used by the tracer.
        Must match the configuration and may alter between different machines.
    """

    name = "Scene Tracer"

    # param struct
    class TraceParams(Structure):
        """Structure matching the shader's trace params"""
        _fields_ = [
            ("targetIdx", c_uint32),
            ("scatterCoefficient", c_float),
            ("maxTime", c_float),
            ("_lowerBBoxCorner", vec3),
            ("_upperBBoxCorner", vec3),
        ]
    class Push(Structure):
        """Structure matching the shader's push constant"""
        _fields_ = [("saveRays", c_int32)]
    
    def __init__(
        self,
        capacity: int,
        scene: Scene,
        *,
        rng: RNG,
        nScattering: int = 6,
        nIterations: int = 1,
        nPhotons: int = 4,
        maxTime: float = 1000.0,
        scatterCoefficient: float = 10.0,
        targetIdx: int = 0,
        clearHitQueue: bool = True,
        keepRays: bool = False,
        blockSize: int = 128,
        code: Optional[bytes] = None,
    ) -> None:
        super().__init__({"TraceParams": self.TraceParams})
        # save params
        self._rng = rng
        self._capacity = capacity
        self._nScattering = nScattering
        self._nIterations = nIterations
        self._nPhotons = nPhotons
        self._scene = scene
        self.setParams(
            targetIdx=targetIdx,
            scatterCoefficient=scatterCoefficient,
            maxTime=maxTime,
            _lowerBBoxCorner=scene.bbox.lowerCorner,
            _upperBBoxCorner=scene.bbox.upperCorner,
        )

        # calculate sizes
        self._groupSize = -(capacity // -blockSize)
        self._maxSamples = capacity * nScattering * nIterations
        # ray queue buffer flag
        self._clearHitQueue = clearHitQueue
        self._keepRays = keepRays
        bufferRayQueue = keepRays or nIterations > 1

        # compile code if needed
        if code is None:
            # create preamble
            preamble = ""
            preamble += f"#define BATCH_SIZE {blockSize}\n"
            preamble += f"#define HIT_QUEUE_SIZE {self._maxSamples}\n"
            preamble += f"#define N_PHOTONS {nPhotons}\n"
            preamble += f"#define N_SCATTER {nScattering}\n"
            preamble += f"#define QUEUE_SIZE {capacity}\n"
            if bufferRayQueue:
                preamble += f"#define OUTPUT_RAYS\n"
            # compile source code
            headers = {"rng.glsl": rng.sourceCode}
            code = compileShader("tracer.scene.glsl", preamble, headers)
        # save code
        self._code = code
        # create program
        self._program = hp.Program(code)

        # create queue items
        rayItem = theia.items.createRayQueueItem(nPhotons)
        hitItem = theia.items.createHitQueueItem(nPhotons)
        # create queues
        if bufferRayQueue:
            self._rayQueue = [QueueTensor(rayItem, capacity) for _ in range(2)]
        else:
            self._rayQueue = [QueueTensor(rayItem, capacity)]
        self._hitQueue = QueueTensor(hitItem, self._maxSamples)
        self._program.bindParams(
            Detectors=scene.detectors,
            Geometries=scene.geometries,
            HitQueueBuffer=self._hitQueue,
            tlas=scene.tlas,
        )
    
    @property
    def capacity(self) -> int:
        """number of light samples processed per run"""
        return self._capacity

    @property
    def code(self) -> Dict[str, bytes]:
        """
        Dictionary containing the compiled source codes used in the pipeline.
        Can be used to cache code allowing to skip the compilation.
        The configuration must match.
        """
        return self._code

    @property
    def nIterations(self) -> int:
        """
        Number of iterations in the tracer. The tracer gets the chance to merge
        diverging execution between iterations, which might improve performance.
        """
        return self._nIterations

    @property
    def nScattering(self) -> int:
        """max number of scattering events per ray"""
        return self._nScattering

    @property
    def nPhotons(self) -> int:
        """number of photons (wavelengths) per light ray/sample"""
        return self._nPhotons

    @property
    def maxSamples(self) -> int:
        """Maximum number of samples produced per batch"""
        return self.capacity * self.nScattering * self.nIterations

    @property
    def rng(self) -> RNG:
        """Generator used for sampling random numbers"""
        return self._rng

    @property
    def rayQueueIn(self) -> hp.Tensor:
        """Tensor containing the ray queue the tracing starts with"""
        return self._rayQueue[0]

    @property
    def rayQueueOut(self) -> Optional[hp.Tensor]:
        """
        Tensor containing the final rays the tracer stopped tracing,
        i.e. the initial rays minus the ones that directly hit the target.
        None, if the tracer does not save the
        """
        if self._keepRays:
            return self._rayQueue[(self._nIterations + 1) % 2]
        else:
            return None
    
    @property
    def scene(self) -> Scene:
        """Scene to trace rays against"""
        return self._scene

    @property
    def keepRays(self) -> bool:
        """
        True, if the tracer saves the final rays the tracer stopped tracing,
        i.e. the initial rays minus the ones that directly hit the target.
        """
        return self._keepRays

    @property
    def clearHitQueue(self) -> bool:
        """Wether to clear the hit queue before each batch."""
        return self._clearHitQueue

    @property
    def hitQueue(self) -> hp.Tensor:
        """Tensor containing the hit queue"""
        return self._hitQueue
    
    def run(self, i: int) -> List[hp.Command]:
        cmd = []
        self._bindParams(self._program, i)
        self.rng.bindParams(self._program, i)
        # clear queue if needed
        if self.clearHitQueue:
            cmd.append(clearQueue(self.hitQueue))
        # check if we need the double buffering
        if not self.keepRays and self.nIterations == 1:
            self._program.bindParams(RayQueueBuffer=self._rayQueue[0])
            cmd.extend(
                [
                    self._program.dispatch(self._groupSize),
                    hp.flushMemory(),
                ]
            )
        else:
            # swap buffers as needed
            for it in range(self.nIterations):
                self._program.bindParams(
                    RayQueueBuffer=self._rayQueue[it % 2],
                    OutRayQueueBuffer=self._rayQueue[(it + 1) % 2],
                )
                save = self.keepRays or it < self.nIterations - 1
                push = self.Push(saveRays=(1 if save else 0))
                cmd.extend(
                    [
                        clearQueue(self._rayQueue[(it + 1) % 2]),
                        self._program.dispatchPush(bytes(push), self._groupSize),
                        hp.flushMemory(),
                    ]
                )
        # done
        return cmd

class WavefrontPathTracer(PipelineStage):
    """
    Path tracer simulating light traveling through media originating at a
    light source and ending at detectors. Traces rays against the geometries
    defined in scene to simulate accurate intersections and obstructions.
    Uses the wavefront approach, which might be faster depending on the scene
    and hardware.

    Parameters
    ----------
    capacity: int
        Maximum number of samples the tracer can process in a single run.
    scene: Scene
        Scene to trace rays against
    rng: RNG
        Generator for sampling random numbers
    nScattering: int, default=6
        Number of scatter events (both surface and volume) to simulate per ray
    nPhotons: int, default=4
        Number of photons (i.e. wavelengths) in single ray
    maxTime: float, default=1000.0
        Max time after which rays will be canceled
    scatteringCoefficient: float, default=10.0
        Scattering coefficient used for sampling. Can be used to tune the time
        distribution of ray hits
    targetIdx: int, default=0
        Id of the detector, the tracer should try to hit.
        Hits on other detectors are ignored to make the estimates simpler.
    blockSize: int, default=128
        Number of threads in a single local work group (block size in CUDA)
    code: { str: bytes }, default={}
        Dictionary containing the compiled shader code used by the tracer.
        Must match the configuration and may alter between different machines.
    """

    name = "Wavefront Path Tracer"

    # params struct
    class Params(Structure):
        _fields_ = [
            ("targetIdx", c_uint32),
            ("scatterCoefficient", c_float),
            ("maxTime", c_float),
            ("_lowerBBoxCorner", vec3),
            ("_upperBBoxCorner", vec3),
        ]

    def __init__(
        self,
        capacity: int,
        scene: Scene,
        nScattering: int = 6,
        *,
        rng: RNG,
        nPhotons: int = 4,
        maxTime: float = 1000.0,
        scatterCoefficient: float = 10.0,
        targetIdx: int,
        blockSize: int = 128,
        code: Dict[str, bytes] = {},
    ) -> None:
        super().__init__({"Params": self.Params})
        # save params
        self._rng = rng
        self._capacity = capacity
        self._nScattering = nScattering
        self._nPhotons = nPhotons
        self._scene = scene  # skip update code from setter
        self.setParams(
            targetIdx=targetIdx,
            scatterCoefficient=scatterCoefficient,
            maxTime=maxTime,
            _lowerBBoxCorner=scene.bbox.glsl.lowerCorner,
            _upperBBoxCorner=scene.bbox.glsl.upperCorner,
        )

        # calculate dispatch size
        self._groupSize = -(capacity // -blockSize)

        # create common preamble
        preamble = ""
        preamble += f"#define QUEUE_SIZE {capacity}\n"
        preamble += f"#define HIT_QUEUE_SIZE {capacity * nScattering}\n"
        preamble += f"#define N_PHOTONS {nPhotons}\n"
        preamble += f"#define LOCAL_SIZE {blockSize}\n\n"
        headers = {"rng.glsl": rng.sourceCode}
        # compile code
        if "trace" not in code:
            code["trace"] = compileShader("wavefront.trace.glsl", preamble, headers)
        if "intersect" not in code:
            code["intersect"] = compileShader(
                "wavefront.intersection.glsl", preamble, headers
            )
        if "volume" not in code:
            code["volume"] = compileShader("wavefront.volume.glsl", preamble, headers)
        if "shadow" not in code:
            code["shadow"] = compileShader("wavefront.shadow.glsl", preamble, headers)
        # save code
        self._code = code

        # create programs
        self._trace = hp.Program(code["trace"])
        self._intersect = hp.Program(code["intersect"])
        self._shadow = hp.Program(code["shadow"])
        self._volume = hp.Program(code["volume"])

        # create queue items
        intersectionItem = theia.items.createIntersectionQueueItem(nPhotons)
        rayItem = theia.items.createRayQueueItem(nPhotons)
        shadowItem = theia.items.createShadowQueueItem(nPhotons)
        volumeItem = theia.items.createVolumeScatterQueueItem(nPhotons)
        hitsItem = theia.items.createHitQueueItem(nPhotons)
        # allocate queues
        self._intersectionQueue = QueueTensor(intersectionItem, capacity)
        self._rayQueue = QueueTensor(rayItem, capacity)
        self._shadowQueue = QueueTensor(shadowItem, capacity)
        self._volumeQueue = QueueTensor(volumeItem, capacity)
        self._hitsQueue = QueueTensor(hitsItem, capacity * nScattering)

        # bind resources to programs
        params = {
            "Detectors": scene.detectors,
            "Geometries": scene.geometries,
            "HitQueueBuffer": self._hitsQueue,
            "IntersectionQueueBuffer": self._intersectionQueue,
            "RayQueueBuffer": self._rayQueue,
            "ShadowQueueBuffer": self._shadowQueue,
            "tlas": scene.tlas,
            "VolumeScatterQueueBuffer": self._volumeQueue,
        }
        self._trace.bindParams(**params)
        self._intersect.bindParams(**params)
        self._shadow.bindParams(**params)
        self._volume.bindParams(**params)

    @property
    def capacity(self) -> int:
        """number of light samples processed per run"""
        return self._capacity

    @property
    def code(self) -> Dict[str, bytes]:
        """
        Dictionary containing the compiled source codes used in the pipeline.
        Can be used to cache code allowing to skip the compilation.
        The configuration must match.
        """
        return self._code

    @property
    def nScattering(self) -> int:
        """max number of scattering events per ray"""
        return self._nScattering

    @property
    def nPhotons(self) -> int:
        """number of photons (wavelengths) per light ray/sample"""
        return self._nPhotons

    @property
    def maxSamples(self) -> int:
        """Maximum number of samples produced per batch"""
        return self.capacity * self.nScattering

    @property
    def rng(self) -> RNG:
        """Generator used for sampling random numbers"""
        return self._rng

    @property
    def scene(self) -> Scene:
        """
        Scene the light rays are traced against.
        Changing the scene only alters new pipelines.
        """
        return self._scene

    @scene.setter
    def scene(self, value: Scene) -> None:
        self._scene = value
        # update params
        self.setParams(
            _lowerBBoxCorner=value.bbox.glsl.lowerCorner,
            _upperBBoxCorner=value.bbox.glsl.upperCorner,
        )
        params = {
            "Detectors": value.detectors,
            "Geometries": value.geometries,
            "tlas": value.tlas,
        }
        self._trace.bindParams(**params)
        self._intersect.bindParams(**params)
        self._shadow.bindParams(**params)
        self._volume.bindParams(**params)

    @property
    def rayQueue(self) -> hp.Tensor:
        """Tensor containing the ray queue"""
        return self._rayQueue

    @property
    def hitsQueue(self) -> hp.Tensor:
        """Tensor containing the hits queue"""
        return self._hitsQueue

    # pipeline stage api

    def run(self, i: int) -> List[hp.Command]:
        self._bindParams(self._trace, i)
        self._bindParams(self._intersect, i)
        self._bindParams(self._shadow, i)
        self._bindParams(self._volume, i)
        self.rng.bindParams(self._trace, i)
        self.rng.bindParams(self._volume, i)
        return [
            # clear queues (init state of ray queue is externally handled)
            # it is enough to only reset the item count (first 4 bytes)
            clearQueue(self._intersectionQueue),
            clearQueue(self._shadowQueue),
            clearQueue(self._volumeQueue),
            # simulate scattering
            *[
                # trace rays
                self._trace.dispatch(self._groupSize),
                clearQueue(self._rayQueue),
                hp.flushMemory(),
                # volume and intersection can work in parallel
                self._volume.dispatch(self._groupSize),
                self._intersect.dispatch(self._groupSize),
                clearQueue(self._volumeQueue),
                clearQueue(self._intersectionQueue),
                hp.flushMemory(),
                # lastly, the shadow queue
                self._shadow.dispatch(self._groupSize),
                clearQueue(self._shadowQueue),
                hp.flushMemory(),
            ]
            * self.nScattering,
        ]
