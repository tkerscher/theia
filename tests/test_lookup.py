import numpy as np
import hephaistos as hp
import theia.lookup
import pytest
from ctypes import Structure, c_int64, c_float
from scipy.interpolate import RegularGridInterpolator


@pytest.fixture(scope="module")
def sampleData1D():
    """Sample data for 1D lookup testing. Shape: (x, f(x), df/fx(x))"""
    x = np.linspace(0.0, 1.0, 100)
    y = np.sin(4 * np.pi * x) * np.exp(-x) + (1.0 - x)
    grad = -np.exp(-x) * (
        np.exp(x) + np.sin(4 * np.pi * x) - 4 * np.pi * np.cos(4 * np.pi * x)
    )
    return np.stack((x, y, grad), axis=-1)


def test_sampleTable1D(sampleData1D):
    # test boundary
    data = np.array([[0.0, 0.0], [10.0, 10.0]])
    sample = theia.lookup.sampleTable1D(data, 10)[1:]  # drop header
    assert abs(sample.min()) < 1e-5
    assert abs(sample.max() < 10.0) < 1e-5
    assert len(sample) == 10
    # test arbitrary boundary
    sample = theia.lookup.sampleTable1D(data, 10, boundary=(5.0, 8.0))[1:]
    assert abs(sample.min() - 5.0) < 1e-5
    assert abs(sample.max() - 8.0) < 1e-5
    assert len(sample) == 10
    # test cubic spline (error should be minimal)
    sample = theia.lookup.sampleTable1D(sampleData1D, 1024)[1:]  # drop header
    x = np.linspace(0.0, 1.0, 1024)
    y = np.sin(4 * np.pi * x) * np.exp(-x) + (1.0 - x)
    assert np.abs(y - sample).max() < 5e-3  # TODO: what is a sensible value?


def test_lookup1D(sampleData1D, shaderUtil):
    N = 8192
    # prepare gpu
    table = hp.FloatTensor(theia.lookup.createTable(sampleData1D[:, 1]))
    tensor = hp.FloatTensor(N)
    buffer = hp.FloatBuffer(N)
    program = shaderUtil.createTestProgram("lookup.test.1D.glsl")
    program.bindParams(OutputBuffer=tensor)
    # run program
    class Push(Structure):
        _fields_ = [("table", c_int64), ("normalization", c_float)]

    push = Push(table=table.address, normalization=(N - 1.0))
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(tensor, buffer))
        .Submit()
        .wait()
    )

    # recreate expected linear interpolation
    x = np.linspace(0, 1.0, N)
    y = np.interp(x, sampleData1D[:, 0], sampleData1D[:, 1])
    # compare
    assert np.abs(y - buffer.numpy()).max() < 1e-6


def test_lookup1Ddx(sampleData1D, shaderUtil):
    N = 8192
    # prepare gpu
    table = hp.FloatTensor(theia.lookup.createTable(sampleData1D[:, 1]))
    valueTensor = hp.FloatTensor(N)
    derivTensor = hp.FloatTensor(N)
    value = hp.FloatBuffer(N)
    deriv = hp.FloatBuffer(N)
    program = shaderUtil.createTestProgram("lookup.test.1Ddx.glsl")
    program.bindParams(ValueOut=valueTensor, DerivOut=derivTensor)
    # run program
    class Push(Structure):
        _fields_ = [("table", c_int64), ("normalization", c_float)]

    push = Push(table=table.address, normalization=(N - 1.0))
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(valueTensor, value))
        .Then(hp.retrieveTensor(derivTensor, deriv))
        .Submit()
        .wait()
    )

    # recreated expected values
    x = np.linspace(0, 1.0, N)
    y = np.interp(x, sampleData1D[:, 0], sampleData1D[:, 1])
    sampleDx = np.gradient(sampleData1D[:, 1], 1.0 / len(sampleData1D))
    grad = np.interp(x, sampleData1D[:, 0], sampleDx)
    # compare
    assert np.abs(y - value.numpy()).max() < 1e-6
    assert np.abs(grad - deriv.numpy()).max() < 0.2
    # unfortunately, numerical derivation on small samples is really this bad...
    # let's check if it is at least not biased
    assert np.mean(grad - deriv.numpy()) < 2e-4


@pytest.fixture
def sampleData2D():
    """Sample data for 2D lookup test. Shape: (x,y,f(x,y))"""
    x = np.linspace(0.0, 1.0, 50)
    y = np.linspace(0.0, 1.0, 50)
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    z = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * (1.0 - y * y) * (1.0 - x * x)
    return np.stack((x, y, z), axis=-1)


def test_sampleTable2D(sampleData2D):
    # test boundary
    data_x = np.array([0.0, 0.0, 10.0, 10.0])
    data_y = np.array([0.0, 10.0, 0.0, 10.0])
    data_z = data_x + data_y
    data = np.stack((data_x, data_y, data_z), axis=-1)
    sample = theia.lookup.sampleTable2D(data, 100, 100, boundaries=(None, (3.0, 8.0)))[
        2:
    ]
    assert len(sample) == 100 * 100
    assert abs(sample[:100].min() - 3) < 1e-5
    assert abs(sample[:100].max() - 13.0) < 1e-5
    assert abs(sample[::100].min() - 3.0) < 1e-5
    assert abs(sample[::100].max() - 8.0) < 1e-5
    # test cubic spline
    sample = theia.lookup.sampleTable2D(sampleData2D, 250, 250, mode="cubic")[2:]
    x = np.linspace(0.0, 1.0, 250)
    y = np.linspace(0.0, 1.0, 250)
    x, y = np.meshgrid(x, y)
    z = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * (1.0 - y * y) * (1.0 - x * x)
    assert np.abs(z.flatten() - sample).max() < 5e-3  # TODO: Find a reasonable value


def test_lookup2D(sampleData2D, shaderUtil):
    N = 256
    # prepare gpu
    table = hp.FloatTensor(theia.lookup.createTable(sampleData2D[:, 2].reshape(50, 50)))
    image = hp.Image(hp.ImageFormat.R32_SFLOAT, N, N)
    buffer = hp.FloatBuffer(N * N)
    program = shaderUtil.createTestProgram("lookup.test.2D.glsl")
    program.bindParams(outputImage=image)
    # run program
    class Push(Structure):
        _fields_ = [("table", c_int64), ("normalization", c_float)]

    push = Push(table=table.address, normalization=(N - 1.0))
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), N // 4, N // 4))
        .Then(hp.retrieveImage(image, buffer))
        .Submit()
        .wait()
    )

    # recreate expected linear interpolation
    x = np.linspace(0.0, 1.0, 50)
    # RegularGridInterpolator wants matrix order ij, instead of xy, thus transpose
    interp = RegularGridInterpolator((x, x), sampleData2D[:, 2].reshape(50, 50).T)
    x = np.linspace(0.0, 1.0, N)
    x, y = np.meshgrid(x, x)
    p = np.stack((x.flatten(), y.flatten()), axis=-1)
    z = interp(p)
    assert np.abs(z - buffer.numpy()).max() < 1e-6


def test_getTableSize(sampleData1D, sampleData2D):
    assert (
        theia.lookup.getTableSize(sampleData1D[:, 1])
        == theia.lookup.createTable(sampleData1D[:, 1]).nbytes
    )
    assert (
        theia.lookup.getTableSize(sampleData2D[:, 2])
        == theia.lookup.createTable(sampleData2D[:, 2]).nbytes
    )
