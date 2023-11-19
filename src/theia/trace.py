import hephaistos as hp

from hephaistos.glsl import vec3
from hephaistos.pipeline import PipelineStage
from hephaistos.queue import QueueTensor

import theia.items
from theia.scene import Scene
from theia.random import RNG
from theia.util import compileShader

from ctypes import Structure, c_float, c_uint32
from typing import Dict, List


class WavefrontPathTracer(PipelineStage):
    """
    Path tracer simulating light traveling through media originating at a
    light source and ending at detectors.

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
            ("lowerBBoxCorner", vec3),
            ("upperBBoxCorner", vec3),
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
        self.maxTime = maxTime
        self.scatterCoefficient = scatterCoefficient
        self.targetIdx = targetIdx
        self._scene = scene  # skip update code from setter

        # calculate dispatch size
        self._groupSize = -(capacity // -blockSize)

        # create common preamble
        preamble = ""
        preamble += f"#define QUEUE_SIZE {capacity}\n"
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
            "IntersectionQueue": self._intersectionQueue,
            "RayQueue": self._rayQueue,
            "ResponseQueue": self._hitsQueue,
            "ShadowQueue": self._shadowQueue,
            "tlas": scene.tlas,
            "VolumeScatterQueue": self._volumeQueue,
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
    def maxTime(self) -> float:
        """upper limit on the simulated elapsed time of photons"""
        return self.getParam("Params", "maxTime")

    @maxTime.setter
    def maxTime(self, value: float) -> None:
        self.setParam("Params", "maxTime", value)

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
        self.setParam("Params", "lowerBBoxCorner", value.bbox.glsl.lowerCorner)
        self.setParam("Params", "upperBBoxCorner", value.bbox.glsl.upperCorner)
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
    def scatteringCoefficient(self) -> float:
        """Scattering coefficient used to sample ray lengths between scatter events"""
        return self.getParam("Params", "scatteringCoefficient")

    @scatteringCoefficient.setter
    def scatteringCoefficient(self, value: float) -> None:
        self.setParam("Params", "scatteringCoefficient", value)

    @property
    def targetIdx(self) -> int:
        """idx of target detector"""
        return self.getParam("Params", "targetIdx")

    @targetIdx.setter
    def targetIdx(self, value: int) -> None:
        self.setParam("Params", "targetIdx", value)

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
            hp.clearTensor(self._intersectionQueue, size=4),
            hp.clearTensor(self._shadowQueue, size=4),
            hp.clearTensor(self._volumeQueue, size=4),
            # simulate scattering
            *[
                # trace rays
                self._trace.dispatch(self._groupSize),
                hp.clearTensor(self._rayQueue, size=4),
                # volume and intersection can work in parallel
                self._volume.dispatch(self._groupSize),
                self._intersect.dispatch(self._groupSize),
                hp.clearTensor(self._volumeQueue, size=4),
                hp.clearTensor(self._intersectionQueue, size=4),
                # lastly, the shadow queue
                self._shadow.dispatch(self._groupSize),
                hp.clearTensor(self._shadowQueue, size=4),
            ]
            * self.nScattering,
        ]
