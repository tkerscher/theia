from __future__ import annotations

from hephaistos.pipeline import Pipeline, PipelineScheduler

from theia.material import MaterialStore, PureWaterModel
from theia.random import PhiloxRNG
from theia.light import SphericalLightSource, UniformWavelengthSource
from theia.response import EmptyResponse
from theia.scene import RectBBox
from theia.target import SphereTarget
from theia.trace import VolumeForwardTracer, VolumePhotonTracer
import theia.units as u

from common import Benchmark


class EmptyVolumePhotonBenchmark(Benchmark, name="Empty Volume Photon Tracer"):
    """
    Simple benchmark for tracing performance. Traces photons in an infinite
    empty detector filled with water.
    """

    def __init__(self) -> None:
        super().__init__()

    def setup(self) -> None:
        # create water medium
        water = PureWaterModel().createMedium()
        self._store = MaterialStore([], media=[water])
        # create tracing pipeline
        rng = PhiloxRNG(key=0xABBA)
        photons = UniformWavelengthSource(lambdaRange=(300.0, 700.0) * u.nm)
        source = SphericalLightSource(budget=1e9)
        # we put the target far away since we want to focus on the tracing part
        target = SphereTarget(position=(1e5, 1e5, 1e5) * u.km)
        response = EmptyResponse()
        tracer = VolumePhotonTracer(
            int(1e7),
            source,
            target,
            photons,
            response,
            rng,
            medium=self._store.media["water"],
            traceBBox=RectBBox((-1e5, -1e5, -1e5) * u.km, (1e5, 1e5, 1e5) * u.km),
            maxTime=float("inf"),
        )
        pipeline = Pipeline(tracer.collectStages())
        self._scheduler = PipelineScheduler(pipeline)

    def run(self):
        tasks = [{} for _ in range(10)]
        self._scheduler.schedule(tasks)
        self._scheduler.wait()

    def finish(self) -> None:
        self._scheduler.destroy()


class EmptyVolumeBenchmark(Benchmark, name="Empty Volume Tracer"):
    """
    Simple benchmark for tracing performance. Traces rays in an infinite empty
    detector filled with water.
    """

    def __init__(self) -> None:
        super().__init__()

    def setup(self) -> None:
        # create water medium
        water = PureWaterModel().createMedium()
        self._store = MaterialStore([], media=[water])
        # create tracing pipeline
        rng = PhiloxRNG(key=0xABBA)
        photons = UniformWavelengthSource(lambdaRange=(300.0, 700.0) * u.nm)
        source = SphericalLightSource(budget=1e9)
        # we put the target far away since we want to focus on the tracing part
        target = SphereTarget(position=(1e5, 1e5, 1e5) * u.km)
        response = EmptyResponse()
        tracer = VolumeForwardTracer(
            int(1e8),
            source,
            target,
            photons,
            response,
            rng,
            nScattering=50,
            medium=self._store.media["water"],
            traceBBox=RectBBox((-1e5, -1e5, -1e5) * u.km, (1e5, 1e5, 1e5) * u.km),
            maxTime=float("inf"),
        )
        pipeline = Pipeline(tracer.collectStages())
        self._scheduler = PipelineScheduler(pipeline)

    def run(self):
        tasks = [{} for _ in range(10)]
        self._scheduler.schedule(tasks)
        self._scheduler.wait()

    def finish(self) -> None:
        self._scheduler.destroy()
