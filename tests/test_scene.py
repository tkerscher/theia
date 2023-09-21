import numpy as np
import hephaistos as hp
import theia.material
import theia.scene
from ctypes import *
from hephaistos.glsl import vec3, stackVector
from numpy.lib.recfunctions import structured_to_unstructured


def test_traverseScene(rng, shaderUtil):
    N = 32 * 1024
    wavelength = 500.0

    class Query(Structure):
        _fields_ = [
            ("position", vec3),
            ("direction", vec3),
            ("log_trans", c_float),
            ("log_prob", c_float),
            ("t0", c_float),
        ]

    class Result(Structure):
        _fields_ = [
            ("position", vec3),
            ("log_trans", c_float),
            ("log_prob", c_float),
            ("t", c_float),
            ("hit", c_float),
        ]

    class Push(Structure):
        _fields_ = [("medium", c_uint64), ("wavelength", c_float)]

    # build material
    class WaterModel(
        theia.material.WaterBaseModel,
        theia.material.HenyeyGreensteinPhaseFunction,
        theia.material.MediumModel,
    ):
        def __init__(self) -> None:
            theia.material.WaterBaseModel.__init__(self, 5.0, 1000.0, 35.0)
            theia.material.HenyeyGreensteinPhaseFunction.__init__(self, 0.6)

        ModelName = "water"

    model = WaterModel()
    water = model.createMedium()
    material = theia.material.Material("material", water, None)
    mat_tensor, mats, media = theia.material.bakeMaterials(material)
    # look up constants
    vg = model.group_velocity(wavelength)
    mu_s = model.scattering_coef(wavelength)
    mu_a = model.absorption_coef(wavelength)
    mu_e = mu_s + mu_a

    # create scene
    store = theia.scene.MeshStore(
        {
            "cone": "assets/cone.stl",
            "cube": "assets/cube.stl",
            "monkey": "assets/suzanne.stl",
        }
    )
    trafo = theia.scene.Transform.Scale(10.0, 10.0, 10.0)
    cube = store.createInstance("cube", "material", transform=trafo)
    scene = theia.scene.Scene([cube], mats, media["water"])

    # reserve memory for i/o
    query_tensor = hp.ArrayTensor(Query, N)
    query_buffer = hp.ArrayBuffer(Query, N)
    result_tensor = hp.ArrayTensor(Result, N)
    result_buffer = hp.ArrayBuffer(Result, N)
    # fill query buffer
    queries = query_buffer.numpy()
    theta = rng.random(N)
    phi = rng.random(N)
    queries["position"] = stackVector(
        (8.0 * rng.random(N), 8.0 * rng.random(N), 8.0 * rng.random(N)), vec3
    )
    queries["direction"] = stackVector(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), vec3
    )
    queries["log_trans"] = rng.random(N)
    queries["log_prob"] = rng.random(N)
    queries["t0"] = 50.0 * rng.random(N)
    # create push constants
    push = Push(medium=media["water"], wavelength=wavelength)

    # create and run test
    program = shaderUtil.createTestProgram("scene.traverse.test.glsl")
    program.bindParams(
        tlas=scene.tlas,
        Scene=scene.scene,
        QueryBuffer=query_tensor,
        ResultBuffer=result_tensor,
    )
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )

    # check result
    results = result_buffer.numpy()
    # check all positions inside cube
    outPos = structured_to_unstructured(results["position"])
    assert np.abs(outPos).max() <= 10.0 + 5e-6
    # check direction
    inPos = structured_to_unstructured(queries["position"])
    dir_in = structured_to_unstructured(queries["direction"])
    distance = np.sqrt(np.square(outPos - inPos).sum(-1))
    dir_res = (outPos - inPos) / distance[:, None]
    assert np.abs(dir_in - dir_res).max() <= 5e-5
    # check transmission
    delta_log_trans = results["log_trans"] - queries["log_trans"]
    assert np.abs(delta_log_trans + mu_e * distance).max() <= 1e-3
    # check hits
    hits_mask = np.abs(np.abs(outPos).max(-1) - 10.0) <= 5e-6
    assert np.all(results["hit"][hits_mask] == 1.0)
    assert np.all(results["hit"][~hits_mask] == -1.0)
    # check prob
    delta_prob = results["log_prob"] - queries["log_prob"]
    assert (
        np.abs(delta_prob[hits_mask] + mu_s * distance[hits_mask]).max() <= 5e-5
    )  # hit
    assert (
        np.abs(
            delta_prob[~hits_mask] - np.log(mu_s) + mu_s * distance[~hits_mask]
        ).max()
        <= 6e-4
    )  # no hit
    # check delta time
    delta_t = results["t"] - queries["t0"]
    assert np.allclose(distance / vg, delta_t, 5e-4)
