import numpy as np
import hephaistos as hp
import os.path
import theia.material
import warnings
from ctypes import Structure, c_uint64, c_float
from scipy.integrate import quad


def test_BK7Model(dataDir, testDataDir):
    model = theia.material.BK7Model()

    n_exp = np.loadtxt(
        os.path.join(testDataDir, "bk7_refractive_index.csv"), delimiter=",", skiprows=2
    )
    assert np.abs(n_exp[:, 1] - model.refractive_index(n_exp[:, 0])).max() < 5e-5

    # numerically derive group velocity to check against
    l = np.linspace(300.0, 800.0, 200)
    n = model.refractive_index(l)
    vg_exp = theia.material.speed_of_light / (
        n - l * np.gradient(n, 500 / (len(l) - 1))
    )
    assert np.abs((vg_exp - model.group_velocity(l)) / vg_exp).max() < 1e-3
    # TODO: finetune limit

    # reuse the transmission data for testing
    trans = np.loadtxt(
        os.path.join(dataDir, "bk7_transmission.csv"), delimiter=",", skiprows=2
    )
    mu_a = model.absorption_coef(trans[:, 0])
    t_10mm = np.exp(-mu_a * 0.01)
    t_25mm = np.exp(-mu_a * 0.025)
    assert np.abs(t_10mm - trans[:, 1]).max() < 0.1  # larger error expected
    assert np.abs(t_10mm - trans[:, 1]).mean() < 0.02  # but on average better
    assert np.abs(t_25mm - trans[:, 2]).max() < 0.01  # larger error expected
    assert np.abs(t_25mm - trans[:, 2]).mean() < 2e-3  # but on average better


def test_WaterBaseModel(dataDir, testDataDir):
    model = theia.material.WaterBaseModel(10.0, 0.0, 35.0)

    # load test data for water
    data = np.loadtxt(
        os.path.join(testDataDir, "water_n_10C_35S.csv"), delimiter=",", skiprows=3
    )
    assert np.abs(data[:, 1] - model.refractive_index(data[:, 0])).max() < 0.005

    # numerically derive group velocity to check against
    l = np.linspace(300.0, 800.0, 200)
    n = model.refractive_index(l)
    vg_exp = theia.material.speed_of_light / (
        n - l * np.gradient(n, 500 / (len(l) - 1))
    )
    assert (
        np.abs((vg_exp - model.group_velocity(l)) / vg_exp).max() < 5e-3
    )  # TODO: finetune limit

    # reuse the table for testing
    data = np.loadtxt(
        os.path.join(dataDir, "water_smith81.csv"), delimiter=",", skiprows=2
    )
    # test against expected data
    assert np.abs(data[:, 1] - model.absorption_coef(data[:, 0])).max() < 1e-6
    assert np.abs(data[:, 2] - model.scattering_coef(data[:, 0])).max() < 1e-6


def getSamplingError(rng, model, bins=50, N=int(1e6)):
    """helper function for testing sampling function"""
    eta = rng.random(N)
    samples = model.phase_sampling(eta)
    h, edges = np.histogram(samples, bins=bins)
    p_bin = h / N

    def f(x):
        return np.exp(model.log_phase_function(x))

    exp_bin = [quad(f, edges[i], edges[i + 1])[0] * 2 * np.pi for i in range(bins)]
    return np.abs(p_bin - exp_bin).max()


def integratePhase(model):
    """Helper function for integrating the log phase function"""

    def f(x):
        return np.exp(model.log_phase_function(x))

    return quad(f, -1.0, 1.0)[0] * 2 * np.pi


def test_HenyeyGreenstein(testDataDir, rng):
    data = np.loadtxt(
        os.path.join(testDataDir, "log_phase_hg.csv"), delimiter=",", skiprows=1
    )
    # we test the model for three different g values
    hg1 = theia.material.HenyeyGreensteinPhaseFunction(0.3)
    hg2 = theia.material.HenyeyGreensteinPhaseFunction(0.0)
    hg3 = theia.material.HenyeyGreensteinPhaseFunction(-0.5)
    hg99 = theia.material.HenyeyGreensteinPhaseFunction(0.99)
    # test phase function
    assert np.abs(data[:, 1] - hg1.log_phase_function(data[:, 0])).max() < 1e-6
    assert np.abs(data[:, 2] - hg2.log_phase_function(data[:, 0])).max() < 1e-6
    assert np.abs(data[:, 3] - hg3.log_phase_function(data[:, 0])).max() < 1e-6
    assert np.abs(data[:, 4] - hg99.log_phase_function(data[:, 0])).max() < 1e-6
    # test sampling function
    assert getSamplingError(rng, hg1) < 5e-4
    assert getSamplingError(rng, hg2) < 5e-4
    assert getSamplingError(rng, hg3) < 5e-4
    assert getSamplingError(rng, hg99) < 5e-4
    # check normalization
    assert abs(integratePhase(hg1) - 1.0) < 1e-5
    assert abs(integratePhase(hg2) - 1.0) < 1e-5
    assert abs(integratePhase(hg3) - 1.0) < 1e-5
    assert abs(integratePhase(hg99) - 1.0) < 1e-5


def test_FournierForand(testDataDir, rng):
    model = theia.material.FournierForandPhaseFunction(1.175, 4.065)
    # load expected data (cos_theta, pdf)
    # note that this was generated from the same code,
    # but was inspected before saving
    data = np.loadtxt(
        os.path.join(testDataDir, "log_phase_ff.csv"), delimiter=",", skiprows=1
    )
    assert np.all(np.abs(data[:, 1] - model.log_phase_function(data[:, 0])) < 1e-6)
    # test sampling function
    # tricky integrand, scipy likely complains -> turn off warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert getSamplingError(rng, model, 70, int(1e7)) < 0.01
    # check normalization
    assert abs(integratePhase(model) - 1.0) < 1e-5


def test_MediumShader(shaderUtil, rng):
    N = 32 * 256

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
    # sample medium for gpu
    buffer = hp.RawBuffer(water.byte_size)
    tensor = hp.ByteTensor(water.byte_size)
    water_adr, *_ = theia.material.serializeMedium(
        water, buffer.address, tensor.address
    )

    # let's define the structs used in the shader
    class Query(Structure):
        _fields_ = [
            ("wavelength", c_float),
            ("theta", c_float),
            ("eta", c_float),
        ]

    class Result(Structure):
        _fields_ = [
            ("n", c_float),
            ("vg", c_float),
            ("mu_s", c_float),
            ("mu_e", c_float),
            ("log_phase", c_float),
            ("angle", c_float),
        ]

    class Push(Structure):
        _fields_ = [("medium", c_uint64)]

    push = Push(medium=water_adr)

    # reserve memory for shader
    query_tensor = hp.ArrayTensor(Query, N)
    query_buffer = hp.ArrayBuffer(Query, N)
    result_tensor = hp.ArrayTensor(Result, N)
    result_buffer = hp.ArrayBuffer(Result, N)
    # fill query structures with random parameters
    queries = query_buffer.numpy()
    delta_lambda = water.lambda_max - water.lambda_min
    queries["wavelength"] = rng.random(N, np.float32) * delta_lambda + water.lambda_min
    queries["theta"] = 1.0 - 2 * rng.random(N, np.float32)  # [-1,1]
    queries["eta"] = rng.random(N, np.float32)  # [0,1]

    # create program
    program = shaderUtil.createTestProgram("medium.test.glsl")
    program.bindParams(QueryBuffer=query_tensor, Results=result_tensor, Foo=tensor)
    # run it
    (
        hp.beginSequence()
        .And(hp.updateTensor(buffer, tensor))  # upload medium
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )

    # recreate expected result
    n = model.refractive_index(queries["wavelength"])
    vg = model.group_velocity(queries["wavelength"])
    mu_a = model.absorption_coef(queries["wavelength"])
    mu_s = model.scattering_coef(queries["wavelength"])
    mu_e = mu_a + mu_s
    log_phase = model.log_phase_function(queries["theta"])
    angle = model.phase_sampling(queries["eta"])
    # test results
    results = result_buffer.numpy()
    assert np.allclose(results["n"], n, 1e-4)
    assert np.allclose(results["vg"], vg, 1e-4)
    assert np.allclose(results["mu_s"], mu_s, 5e-3)
    assert np.allclose(results["mu_e"], mu_e, 5e-3)
    # assert np.allclose(results["log_phase"], log_phase, 1e-4)
    assert np.abs(results["log_phase"] - log_phase).max() < 5e-5
    assert np.allclose(results["angle"], angle, 1e-4, 1e-5)


def test_MaterialShader(shaderUtil, rng):
    N = 32 * 1000  # amount of samples/shader calls
    # we start with building some simple materials
    class WaterModel(
        theia.material.WaterBaseModel,
        theia.material.HenyeyGreensteinPhaseFunction,
        theia.material.MediumModel,
    ):
        def __init__(self) -> None:
            theia.material.WaterBaseModel.__init__(self, 5.0, 1000.0, 35.0)
            theia.material.HenyeyGreensteinPhaseFunction.__init__(self, 0.6)

        ModelName = "water"

    water_model = WaterModel()
    water = water_model.createMedium()
    glass_model = theia.material.BK7Model()
    glass = glass_model.createMedium(name="glass", num_lambda=4096)
    mat_water_glass = theia.material.Material("water_glass", water, glass)
    mat_vac_glass = theia.material.Material("vac_glass", None, glass)
    # bake materials
    tensor, mats, media = theia.material.bakeMaterials(mat_water_glass, mat_vac_glass)

    # let's define the structs used in the shader
    class Query(Structure):
        _fields_ = [
            ("material", c_uint64),
            ("lam", c_float),  # wavelength
            ("theta", c_float),
            ("eta", c_float),
            ("padding", c_float),
        ]

    class Result(Structure):
        _fields_ = [
            ("n", c_float),
            ("vg", c_float),
            ("mu_s", c_float),
            ("mu_e", c_float),
            ("log_phase", c_float),
            ("angle", c_float),
        ]

    # reserve memory for shader
    query_tensor = hp.ArrayTensor(Query, N)
    query_buffer = hp.ArrayBuffer(Query, N)
    result_tensor = hp.ArrayTensor(Result, 2 * N)  # both inside and outside
    result_buffer = hp.ArrayBuffer(Result, 2 * N)
    # fill query structures with random parameters
    queries = query_buffer.numpy()
    queries["material"] = [mats["water_glass"],] * (N // 2) + [
        mats["vac_glass"],
    ] * (N // 2)
    queries["lam"] = rng.random(N, np.float32) * 600.0 + 200.0  # [200,800]nm
    queries["theta"] = 1.0 - 2 * rng.random(N, np.float32)  # [-1,1]
    queries["eta"] = rng.random(N, np.float32)  # [0,1]

    # create program
    program = shaderUtil.createTestProgram("material.test.glsl")
    program.bindParams(QueryBuffer=query_tensor, Results=result_tensor)
    # run it
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .Submit()
        .wait()
    )

    # recreate expected result
    gpu = result_buffer.numpy()
    cpu = np.empty((N * 2, 6))
    # refractive index
    cpu[:N:2, 0] = water_model.refractive_index(queries["lam"][: N // 2])
    cpu[1:N:2, 0] = glass_model.refractive_index(queries["lam"][: N // 2])
    cpu[N::2, 0] = 1.0  # vacuum
    cpu[N + 1 :: 2, 0] = glass_model.refractive_index(queries["lam"][N // 2 :])
    # group velocity
    cpu[:N:2, 1] = water_model.group_velocity(queries["lam"][: N // 2])
    cpu[1:N:2, 1] = glass_model.group_velocity(queries["lam"][: N // 2])
    cpu[N::2, 1] = theia.material.speed_of_light  # vacuum
    cpu[N + 1 :: 2, 1] = glass_model.group_velocity(queries["lam"][N // 2 :])
    # scattering coef
    cpu[:N:2, 2] = water_model.scattering_coef(queries["lam"][: N // 2])
    cpu[1:N:2, 2] = 0.0  # glass does not (volume) scatter
    cpu[N::2, 2] = 0.0  # vacuum
    cpu[N + 1 :: 2, 2] = 0.0  # glass does not (volume) scatter
    # absorption coef
    cpu[:N:2, 3] = water_model.absorption_coef(queries["lam"][: N // 2])
    cpu[1:N:2, 3] = glass_model.absorption_coef(queries["lam"][: N // 2])
    cpu[N::2, 3] = 0.0  # vacuum
    cpu[N + 1 :: 2, 3] = glass_model.absorption_coef(queries["lam"][N // 2 :])
    # extinction coef
    cpu[:, 3] += cpu[:, 2]
    # phase
    cpu[:N:2, 4] = water_model.log_phase_function(queries["theta"][: N // 2])
    cpu[1:N:2, 4] = 0.0  # glass does not (volume) scatter
    cpu[N::2, 4] = 0.0  # vacuum
    cpu[N + 1 :: 2, 4] = 0.0  # glass does not (volume) scatter
    # angle
    cpu[:N:2, 5] = water_model.phase_sampling(queries["eta"][: N // 2])
    cpu[1:N:2, 5] = 0.0  # glass does not (volume) scatter
    cpu[N::2, 5] = 0.0  # vacuum
    cpu[N + 1 :: 2, 5] = 0.0  # glass does not (volume) scatter

    # check result
    assert np.allclose(gpu["n"], cpu[:, 0], 1e-4)
    assert np.allclose(gpu["vg"], cpu[:, 1], 1e-4)
    assert np.allclose(np.exp(-gpu["mu_s"]), np.exp(-cpu[:, 2]), 1e-5, 1e-3)
    assert np.allclose(np.exp(-gpu["mu_e"]), np.exp(-cpu[:, 3]), 1e-5, 1e-3)
    # assert np.allclose(gpu["log_phase"], cpu[:, 4], 1e-4)
    assert np.abs(gpu["log_phase"] - cpu[:, 4]).max() < 5e-4
    assert np.allclose(gpu["angle"], cpu[:, 5], 1e-4, 1e-5)
