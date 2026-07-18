import pytest

import numpy as np
import hephaistos as hp
import os.path
import warnings

from hephaistos.queue import QueueBuffer, QueueTensor
from ctypes import c_float
from scipy.integrate import quad

from theia.compiler import compileShader
from theia.lookup import Table
from theia.model import PureWaterModel
from theia.property import FloatProperty, TableProperty
from theia.util import createCType

import theia.material
import theia.surface
import theia.volume
import theia.units as u


def test_checkMedium():
    medium = PureWaterModel().createMedium()
    assert theia.material.checkMedium(medium)
    # TODO: Ideally, we would check if faulty media would trigger this,
    #       but that's for a later day (maybe)


def test_BK7Model(dataDir, testDataDir):
    model = theia.model.BK7Model()

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


def test_RayleighScattering():
    class Model(
        theia.model.DispersionFreeMedium,
        theia.model.RayleighScatteringModel,
    ):
        def __init__(self):
            super().__init__(
                n=1.4,
                ng=1.4,
                temperature=10.0,
                betaT=4.6e-10,
                depolarizationRatio=0.039,
            )

    assert theia.material.checkMedium(Model().createMedium())


def test_WaterBaseModel(testDataDir):
    model = theia.model.WaterBaseModel(10.0, 0.0, 35.0)

    # load test data for water
    data = np.loadtxt(
        os.path.join(testDataDir, "water_n_10C_35S.csv"), delimiter=",", skiprows=3
    )
    assert np.abs(data[:, 1] - model.refractive_index(data[:, 0] * u.nm)).max() < 0.005
    assert theia.material.checkMedium(model.createMedium())


def test_PureWaterModel():
    model = theia.model.PureWaterModel(
        temperature=5.0,
        pressure=2000.0,
        salinity=35.0,
    )
    assert theia.material.checkMedium(model.createMedium())


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
    hg1 = theia.model.HenyeyGreensteinPhaseFunction(0.3)
    hg2 = theia.model.HenyeyGreensteinPhaseFunction(0.0)
    hg3 = theia.model.HenyeyGreensteinPhaseFunction(-0.5)
    hg99 = theia.model.HenyeyGreensteinPhaseFunction(0.99)
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
    model = theia.model.FournierForandPhaseFunction(1.175, 4.065)
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


def test_MaterialStore():
    N = 32 * 64
    lam_range = (400.0, 700.0) * u.nm

    # create material store
    water_model = PureWaterModel()
    water = water_model.createMedium(wavelengthRange=lam_range)
    glass_model = theia.model.BK7Model()
    glass = glass_model.createMedium(
        name="glass", numSamples=4096, wavelengthRange=lam_range
    )
    f = ("TRVB", "BDL")
    surface_model = theia.surface.DielectricSurface()
    mat_water_glass = theia.material.Material(
        "water_glass", water, glass, surface_model, flags=f
    )
    mat_vac_glass = theia.material.Material("vac_glass", None, glass, surface_model)
    mat_store = theia.material.MaterialStore([mat_vac_glass, mat_water_glass])

    # allocate memory
    queue_fields = [
        ("u", c_float),
        ("i_n", c_float),
        ("i_vg", c_float),
        ("i_mu_s", c_float),
        ("i_log_phase", c_float),
        ("i_angle", c_float),
        ("o_n", c_float),
        ("o_vg", c_float),
        ("o_mu_s", c_float),
        ("o_log_phase", c_float),
        ("o_angle", c_float),
    ]
    queueItem = createCType("Item", queue_fields)
    queue_tensor = QueueTensor(queueItem, N, skipCounter=True)
    queue_buffer = QueueBuffer(queueItem, N, skipCounter=True)
    flag_tensor = hp.UnsignedIntTensor(2)
    flag_buffer = hp.UnsignedIntBuffer(2)

    # create test program
    program = hp.Program(compileShader("material.test.glsl", headers=mat_store.header))
    program.bindParams(Result=queue_tensor, Flags=flag_tensor)
    mat_store.bindParams(program)
    # run it
    (
        hp.beginSequence()
        .And(program.dispatch(32))
        .Then(hp.retrieveTensor(queue_tensor, queue_buffer))
        .And(hp.retrieveTensor(flag_tensor, flag_buffer))
        .Submit()
        .wait()
    )

    # check result
    result = queue_buffer.view
    v = result["u"]
    assert np.all((v >= 0.0) & (v < 1.0))
    lam = v * (lam_range[1] - lam_range[0]) + lam_range[0]
    theta = 2.0 * v - 1.0
    # test inside medium
    assert np.allclose(result["i_n"], water_model.refractive_index(lam))
    assert np.allclose(result["i_vg"], water_model.group_velocity(lam))
    assert np.allclose(result["i_mu_s"], water_model.scattering_coef(lam))
    assert np.allclose(result["i_log_phase"], water_model.log_phase_function(theta))
    assert np.allclose(result["i_angle"], water_model.phase_sampling(v), atol=1e-6)
    # test outside medium
    assert np.allclose(result["o_n"], glass_model.refractive_index(lam))
    assert np.allclose(result["o_vg"], glass_model.group_velocity(lam))
    assert np.allclose(result["o_mu_s"], -10.0)  # glass does not scatter
    assert np.allclose(result["o_log_phase"], -10.0)
    assert np.allclose(result["o_angle"], -10.0)
    # test flags
    assert flag_buffer.numpy()[0] == mat_water_glass.flagsInward
    assert flag_buffer.numpy()[1] == mat_water_glass.flagsOutward


def test_serializeMaterial(tmp_path, rng):
    """create some dummy media/material and try to save/load them"""
    # create dummy media
    transparent = theia.volume.Transparent()
    attenuating = theia.volume.Attenuating(absorb=True)
    rand = lambda n: rng.random(n, dtype=np.float32)
    table = lambda n, a: TableProperty(Table(rand(n) * a))
    med1 = theia.material.Medium(
        "med1",
        (450.0, 750.0),
        {
            "refractice_index": table(456, 2.0),
            "group_velocity": table(196, 0.8),
            "scattering_coef": table(199, 30.0),
            "phase_sampling": table(648, 3.14),
        },
        transparent,
    )
    med2 = theia.material.Medium(
        "med2",
        (500.0, 700.0),
        {
            "refractive_index": table(199, 4.5),
            "group_velocity": table(256, 0.8),
            "absorption_coef": table(145, 25.0),
            "scattering_coef": table(263, 25.0),
            "log_phase_function": table(105, 1.0),
            "phase_sampling": table(156, 1.0),
        },
        attenuating,
    )
    medExtra = theia.material.Medium(
        "extra",
        (450.0, 650.0),
        {
            "refractive_index": table(199, 2.5),
            "group_velocity": table(156, 0.8),
            "absorption_coef": table(133, 25.0),
            "scattering_coef": table(119, 25.0),
            "log_phase_function": table(107, 1.0),
            "phase_sampling": table(108, 1.0),
        },
        attenuating,
    )
    # dummy material
    des = theia.surface.DielectricSurface()
    mat1 = theia.material.Material("mat1", "med1", med2, des, flags=("RDt", "Tfb"))
    mat2 = theia.material.Material("mat2", med1, None, des, flags=("BL", "RbTf"))
    mat2.properties = {
        "albedo": FloatProperty((1.0, 0.5)),
        "anisotropy": table(165, 1.0),
        "ref": theia.material.MediumReferenceProperty(med2),
    }

    # save
    path = tmp_path.joinpath("materials.zip")
    theia.material.saveMaterials(path, [mat1, mat2], media=[medExtra])
    # load
    mat, med = theia.material.loadMaterials(path)

    # check if media was restored correctly
    def checkMedium(test: theia.material.Medium, true: theia.material.Medium):
        assert test.name == true.name
        assert test.wavelengthRange == true.wavelengthRange
        assert test.properties.keys() == true.properties.keys()
        assert test.physicModel == true.physicModel
        for name, prop in true.properties.items():
            # for now we only do table properties
            test_prop = test.properties[name]
            assert isinstance(test_prop, TableProperty)
            assert test_prop.table is not None
            assert isinstance(prop, TableProperty)
            assert prop.table is not None
            assert np.all(test_prop.table.samples == prop.table.samples)

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
    assert mat["mat1"].physicModel == mat1.physicModel
    assert mat["mat2"].inside == med["med1"]
    assert mat["mat2"].outside == None
    assert mat["mat2"].flagsInward == mat2.flagsInward
    assert mat["mat2"].flagsOutward == mat2.flagsOutward
    assert mat["mat2"].physicModel == mat2.physicModel
    assert mat["mat2"].properties.keys() == mat2.properties.keys()
    assert mat["mat2"].properties["albedo"].value == mat2.properties["albedo"].value
    assert mat["mat2"].properties["ref"].medium == "med2"
    t1 = mat["mat2"].properties["anisotropy"].table.samples
    t2 = mat2.properties["anisotropy"].table.samples
    assert np.all(t1 == t2)
