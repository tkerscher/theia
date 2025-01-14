import numpy as np
import hephaistos as hp
import os.path
import theia.material
import theia.units as u
import warnings
from ctypes import Structure, c_uint64, c_float
from scipy.integrate import quad


def test_BK7Model(dataDir, testDataDir):
    model = theia.material.BK7Model()

    n_exp = np.loadtxt(
        os.path.join(testDataDir, "bk7_refractive_index.csv"), delimiter=",", skiprows=2
    )
    assert np.abs(n_exp[:, 1] - model.refractive_index(n_exp[:, 0] * u.nm)).max() < 5e-5

    # numerically derive group velocity to check against
    l = np.linspace(300.0, 800.0, 200) * u.nm
    n = model.refractive_index(l)
    vg_exp = 1.0 / (n - l * np.gradient(n, 500 / (len(l) - 1))) * u.c
    assert np.abs((vg_exp - model.group_velocity(l)) / vg_exp).max() < 1e-3
    # TODO: finetune limit

    # reuse the transmission data for testing
    trans = np.loadtxt(
        os.path.join(dataDir, "bk7_transmission.csv"), delimiter=",", skiprows=2
    )
    mu_a = model.absorption_coef(trans[:, 0] * u.nm)
    t_10mm = np.exp(-mu_a * 10.0 * u.mm)
    t_25mm = np.exp(-mu_a * 25.0 * u.mm)
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
    assert np.abs(data[:, 1] - model.refractive_index(data[:, 0] * u.nm)).max() < 0.005

    # numerically derive group velocity to check against
    l = np.linspace(300.0, 800.0, 200) * u.nm
    n = model.refractive_index(l)
    vg_exp = 1.0 / (n - l * np.gradient(n, 500 / (len(l) - 1))) * u.c
    assert (
        np.abs((vg_exp - model.group_velocity(l)) / vg_exp).max() < 5e-3
    )  # TODO: finetune limit

    # reuse the table for testing
    data = np.loadtxt(
        os.path.join(dataDir, "water_smith81.csv"), delimiter=",", skiprows=2
    )
    # test against expected data
    assert np.abs(data[:, 1] - model.absorption_coef(data[:, 0] * u.nm)).max() < 1e-6
    assert np.abs(data[:, 2] - model.scattering_coef(data[:, 0] * u.nm)).max() < 1e-6


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
    store = theia.material.MaterialStore([], media=[water])

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

    push = Push(medium=store.media["water"])

    # reserve memory for shader
    query_tensor = hp.ArrayTensor(Query, N)
    query_buffer = hp.ArrayBuffer(Query, N)
    result_tensor = hp.ArrayTensor(Result, N)
    result_buffer = hp.ArrayBuffer(Result, N)
    # fill query structures with random parameters
    queries = query_buffer.numpy()
    dLam = water.lambda_max - water.lambda_min
    queries["wavelength"] = (rng.random(N, np.float32) * dLam + water.lambda_min) * u.nm
    queries["theta"] = 1.0 - 2 * rng.random(N, np.float32)  # [-1,1]
    queries["eta"] = rng.random(N, np.float32)  # [0,1]

    # create program
    program = shaderUtil.createTestProgram("medium.test.glsl")
    program.bindParams(QueryBuffer=query_tensor, Results=result_tensor)
    # run it
    (
        hp.beginSequence()
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
        theia.material.KokhanovskyOceanWaterPhaseMatrix,
        theia.material.MediumModel,
    ):
        def __init__(self) -> None:
            theia.material.WaterBaseModel.__init__(self, 5.0, 1000.0, 35.0)
            theia.material.HenyeyGreensteinPhaseFunction.__init__(self, 0.6)
            theia.material.KokhanovskyOceanWaterPhaseMatrix.__init__(
                self, p90=0.66, theta0=0.25, alpha=4.0, xi=25.6  # voss measurement fit
            )

        ModelName = "water"

    water_model = WaterModel()
    water = water_model.createMedium()
    glass_model = theia.material.BK7Model()
    glass = glass_model.createMedium(name="glass", num_lambda=4096)
    mat_water_glass = theia.material.Material("water_glass", water, glass)
    mat_vac_glass = theia.material.Material("vac_glass", None, glass)
    mat_store = theia.material.MaterialStore([mat_water_glass, mat_vac_glass])

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
            ("m12", c_float),
            ("m22", c_float),
            ("m33", c_float),
            ("m34", c_float),
        ]

    # reserve memory for shader
    query_tensor = hp.ArrayTensor(Query, N)
    query_buffer = hp.ArrayBuffer(Query, N)
    result_tensor = hp.ArrayTensor(Result, 2 * N)  # both inside and outside
    result_buffer = hp.ArrayBuffer(Result, 2 * N)
    flag_tensor = hp.UnsignedIntTensor(2)
    flag_buffer = hp.UnsignedIntBuffer(2)
    # fill query structures with random parameters
    queries = query_buffer.numpy()
    queries["material"] = [
        mat_store.material["water_glass"],
    ] * (N // 2) + [
        mat_store.material["vac_glass"],
    ] * (N // 2)
    queries["lam"] = (rng.random(N, np.float32) * 600.0 + 200.0) * u.nm
    queries["theta"] = 1.0 - 2 * rng.random(N, np.float32)  # [-1,1]
    queries["eta"] = rng.random(N, np.float32)  # [0,1]

    # create program
    program = shaderUtil.createTestProgram("material.test.glsl")
    program.bindParams(
        QueryBuffer=query_tensor, Results=result_tensor, Flags=flag_tensor
    )
    # run it
    (
        hp.beginSequence()
        .And(hp.updateTensor(query_buffer, query_tensor))
        .Then(program.dispatch(N // 32))
        .Then(hp.retrieveTensor(result_tensor, result_buffer))
        .And(hp.retrieveTensor(flag_tensor, flag_buffer))
        .Submit()
        .wait()
    )

    # recreate expected result
    gpu = result_buffer.numpy()
    cpu = np.empty((N * 2, 10))
    # refractive index
    cpu[:N:2, 0] = water_model.refractive_index(queries["lam"][: N // 2])
    cpu[1:N:2, 0] = glass_model.refractive_index(queries["lam"][: N // 2])
    cpu[N::2, 0] = 1.0  # vacuum
    cpu[N + 1 :: 2, 0] = glass_model.refractive_index(queries["lam"][N // 2 :])
    # group velocity
    cpu[:N:2, 1] = water_model.group_velocity(queries["lam"][: N // 2])
    cpu[1:N:2, 1] = glass_model.group_velocity(queries["lam"][: N // 2])
    cpu[N::2, 1] = 1.0 * u.c  # vacuum
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
    # m12
    cpu[:N:2, 6] = water_model.phase_m12(queries["theta"][: N // 2])
    cpu[1:N:2, 6] = 0.0  # glass does not (volume) scatter
    cpu[N::2, 6] = 0.0  # vacuum
    cpu[N + 1 :: 2, 6] = 0.0  # glass does not (volume) scatter
    # m22
    cpu[:N:2, 7] = water_model.phase_m22(queries["theta"][: N // 2])
    cpu[1:N:2, 7] = 0.0  # glass does not (volume) scatter
    cpu[N::2, 7] = 0.0  # vacuum
    cpu[N + 1 :: 2, 7] = 0.0  # glass does not (volume) scatter
    # m33
    cpu[:N:2, 8] = water_model.phase_m33(queries["theta"][: N // 2])
    cpu[1:N:2, 8] = 0.0  # glass does not (volume) scatter
    cpu[N::2, 8] = 0.0  # vacuum
    cpu[N + 1 :: 2, 8] = 0.0  # glass does not (volume) scatter
    # m34
    cpu[:N:2, 9] = 0.0  # no m34 for water
    cpu[1:N:2, 9] = 0.0  # glass does not (volume) scatter
    cpu[N::2, 9] = 0.0  # vacuum
    cpu[N + 1 :: 2, 9] = 0.0  # glass does not (volume) scatter

    # check result
    assert np.allclose(gpu["n"], cpu[:, 0], 1e-4)
    assert np.allclose(gpu["vg"], cpu[:, 1], 1e-4)
    assert np.allclose(np.exp(-gpu["mu_s"]), np.exp(-cpu[:, 2]), 1e-5, 1e-3)
    assert np.allclose(np.exp(-gpu["mu_e"]), np.exp(-cpu[:, 3]), 1e-5, 1e-3)
    # assert np.allclose(gpu["log_phase"], cpu[:, 4], 1e-4)
    assert np.abs(gpu["log_phase"] - cpu[:, 4]).max() < 5e-4
    assert np.allclose(gpu["angle"], cpu[:, 5], 1e-4, 1e-5)
    assert np.abs(gpu["m12"] - cpu[:, 6]).max() < 1e-3
    assert np.abs(gpu["m22"] - cpu[:, 7]).max() < 1e-3
    assert np.abs(gpu["m33"] - cpu[:, 8]).max() < 1e-3
    assert np.abs(gpu["m34"] - cpu[:, 9]).max() < 1e-3
    assert flag_buffer.numpy()[0] == mat_vac_glass.flagsInward
    assert flag_buffer.numpy()[1] == mat_vac_glass.flagsOutward


def test_serializeMaterial(tmp_path, rng):
    """create some dummy media/material and try to save/load them"""
    # create dummy media
    med1 = theia.material.Medium(
        "med1",
        450.0,
        750.0,
        refractive_index=rng.random(456, dtype=np.float32) * 2.0,
        group_velocity=rng.random(196, dtype=np.float32) * 0.8,
        absorption_coef=rng.random(233, dtype=np.float32) * 30.0,
        scattering_coef=rng.random(199, dtype=np.float32) * 30.0,
        log_phase_function=rng.random(156, dtype=np.float32) * 1.0,
        phase_sampling=rng.random(648, dtype=np.float32) * 3.14,
        phase_m12=rng.random(354, dtype=np.float32),
        phase_m22=rng.random(416, dtype=np.float32),
        phase_m33=rng.random(422, dtype=np.float32),
        phase_m34=rng.random(235, dtype=np.float32),
    )
    med2 = theia.material.Medium(
        "med2",
        500.0,
        700.0,
        refractive_index=rng.random(199, dtype=np.float32) * 4.5,
        group_velocity=rng.random(256, dtype=np.float32) * 0.8,
        absorption_coef=rng.random(145, dtype=np.float32) * 25.0,
        scattering_coef=rng.random(263, dtype=np.float32) * 25.0,
        log_phase_function=rng.random(105, dtype=np.float32),
        phase_sampling=rng.random(156, dtype=np.float32),
    )
    medExtra = theia.material.Medium(
        "extra",
        450.0,
        650.0,
        refractive_index=rng.random(199, dtype=np.float32),
        group_velocity=rng.random(156, dtype=np.float32) * 0.8,
        absorption_coef=rng.random(133, dtype=np.float32) * 25.0,
        scattering_coef=rng.random(119, dtype=np.float32) * 25.0,
        log_phase_function=rng.random(107, dtype=np.float32),
        phase_sampling=rng.random(108, dtype=np.float32),
    )
    # dummy material
    mat1 = theia.material.Material("mat1", "med1", med2, flags=("R", "Tfb"))
    mat2 = theia.material.Material("mat2", med1, None, flags=("B", "RbTf"))

    # save
    path = tmp_path.joinpath("materials.zip")
    theia.material.saveMaterials(path, [mat1, mat2], media=[medExtra])
    # load
    mat, med = theia.material.loadMaterials(path)

    # check if media was restored correctly
    def checkMedium(test: theia.material.Medium, true: theia.material.Medium):
        assert test.lambda_min == true.lambda_min
        assert test.lambda_max == true.lambda_max
        assert np.all(test.refractive_index == true.refractive_index)
        assert np.all(test.group_velocity == true.group_velocity)
        assert np.all(test.absorption_coef == true.absorption_coef)
        assert np.all(test.scattering_coef == true.scattering_coef)
        assert np.all(test.log_phase_function == true.log_phase_function)
        assert np.all(test.phase_sampling == true.phase_sampling)
        assert np.all(test.phase_m12 == true.phase_m12)
        assert np.all(test.phase_m22 == true.phase_m22)
        assert np.all(test.phase_m33 == true.phase_m33)
        assert np.all(test.phase_m34 == true.phase_m34)

    assert med.keys() == {"med1", "med2", "extra"}
    checkMedium(med["med1"], med1)
    checkMedium(med["med2"], med2)
    checkMedium(med["extra"], medExtra)
    # check if materials were restored correctly
    assert mat.keys() == {"mat1", "mat2"}
    assert mat["mat1"].inside == med["med1"]
    assert mat["mat1"].outside == med["med2"]
    assert mat["mat1"].flagsInward == mat1.flagsInward
    assert mat["mat1"].flagsOutward == mat1.flagsOutward
    assert mat["mat2"].inside == med["med1"]
    assert mat["mat2"].outside == None
    assert mat["mat2"].flagsInward == mat2.flagsInward
    assert mat["mat2"].flagsOutward == mat2.flagsOutward
