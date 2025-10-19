from __future__ import annotations

from pathlib import Path

from hephaistos.pipeline import Pipeline, PipelineScheduler

from theia.material import BK7Model, Material, MaterialStore, PureWaterModel
from theia.light import SphericalLightSource, UniformWavelengthSource
from theia.random import PhiloxRNG
from theia.response import EmptyResponse
from theia.scene import MeshStore, Scene, Transform
from theia.trace import SceneForwardTracer, ScenePhotonTracer
import theia.units as u

from common import Benchmark


class EmptySceneBenchmark(Benchmark, skip=True):
    """Base class creating an empty scene, i.e. no detector"""

    def __init__(self) -> None:
        super().__init__()

    def setup(self) -> None:
        # create materials
        water = PureWaterModel().createMedium()
        glass = BK7Model().createMedium()
        water_glass = Material("water_glass", glass, water)
        water_air = Material("water_air", None, water)
        water_refl = Material("water_refl", water, None, flags="R")
        self._store = MaterialStore([water_air, water_glass, water_refl])
        # create scene
        sphere_path = str(Path(__file__).parent.joinpath("../assets/sphere.stl"))
        self._meshes = MeshStore({"sphere": sphere_path})
        # we will create a large reflective sphere filled with water
        # inside there are several air bubbles and glass spheres
        spheres = [
            # material, radius, x, y, z
            ("water_refl", 30.0 * u.m, 0.0, 0.0, 0.0),
            ("water_glass", *(10.0, 4.125, 5.285, 6.355) * u.m),
            ("water_glass", *(6.5, -6.7566, 0.49569, -11.067) * u.m),
            ("water_glass", *(3.5, -12.773, 0.60137, 0.70611) * u.m),
            ("water_glass", *(5.0, -2.3176, -11.489, 1.2126) * u.m),
            ("water_glass", *(2.8, 10.361, -9.5025, -2.0454) * u.m),
            ("water_air", *(1.0, -5.9025, -3.166, -1.588) * u.m),
            ("water_air", *(3.2, 0.0975, -11.713, -10.835) * u.m),
            ("water_air", *(1.4, 7.0859, -3.166, -10.168) * u.m),
            ("water_air", *(2.9, -9.834, -5.0985, 8.452) * u.m),
        ]
        instances = []
        for material, r, x, y, z in spheres:
            T = Transform.TRS(scale=r, translate=(x, y, z))
            inst = self._meshes.createInstance("sphere", material, T)
            instances.append(inst)
        self._scene = Scene(
            instances, self._store.material, medium=self._store.media["water"]
        )


class ScenePhotonBenchmark(EmptySceneBenchmark, name="Scene Photon Tracer"):
    """
    Simple benchmark for tracing performance. Traces photon in an reflective
    scene without detectors, filled with spheres of different media.
    """

    def __init__(self) -> None:
        super().__init__()

    def setup(self) -> None:
        super().setup()

        # create pipeline
        rng = PhiloxRNG(key=0xABBA)
        photons = UniformWavelengthSource(lambdaRange=(300.0, 700.0) * u.nm)
        source = SphericalLightSource(position=(-11.0, 9.0, 0.0) * u.m, budget=1e9)
        response = EmptyResponse()
        tracer = ScenePhotonTracer(
            int(1e7),
            source,
            photons,
            response,
            rng,
            self._scene,
            maxTime=float("inf"),
        )
        pipeline = Pipeline(tracer.collectStages())
        self._scheduler = PipelineScheduler(pipeline)

    def run(self) -> None:
        tasks = [{} for _ in range(10)]
        self._scheduler.schedule(tasks)
        self._scheduler.wait()

    def finish(self) -> None:
        self._scheduler.destroy()


class SceneTracerBenchmark(EmptySceneBenchmark, name="Scene Tracer"):
    """
    Simple benchmark for tracing performance. Traces rays in an reflective
    scene without detectors, filled with spheres of different media.
    """

    def __init__(self) -> None:
        super().__init__()

    def setup(self) -> None:
        super().setup()

        # create pipeline
        rng = PhiloxRNG(key=0xABBA)
        photons = UniformWavelengthSource(lambdaRange=(300.0, 700.0) * u.nm)
        source = SphericalLightSource(position=(-11.0, 9.0, 0.0) * u.m, budget=1e9)
        response = EmptyResponse()
        tracer = SceneForwardTracer(
            int(1e8),
            source,
            photons,
            response,
            rng,
            self._scene,
            maxPathLength=60,
            maxTime=float("inf"),
        )
        pipeline = Pipeline(tracer.collectStages())
        self._scheduler = PipelineScheduler(pipeline)

    def run(self) -> None:
        tasks = [{} for _ in range(8)]
        self._scheduler.schedule(tasks)
        self._scheduler.wait()

    def finish(self) -> None:
        self._scheduler.destroy()
