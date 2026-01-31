import numpy as np
import hephaistos as hp
import theia.lookup
import pytest
from ctypes import Structure, c_int64, c_float
from scipy.interpolate import RegularGridInterpolator
from theia.compiler import compileShader


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
    sample = theia.lookup.sampleTable1D(data, 10).samples
    assert abs(sample.min()) < 1e-5
    assert abs(sample.max() < 10.0) < 1e-5
    assert len(sample) == 10
    # test arbitrary boundary
    sample = theia.lookup.sampleTable1D(data, 10, boundary=(5.0, 8.0)).samples
    assert abs(sample.min() - 5.0) < 1e-5
    assert abs(sample.max() - 8.0) < 1e-5
    assert len(sample) == 10
    # test cubic spline (error should be minimal)
    sample = theia.lookup.sampleTable1D(sampleData1D, 1024).samples
    x = np.linspace(0.0, 1.0, 1024)
    y = np.sin(4 * np.pi * x) * np.exp(-x) + (1.0 - x)
    assert np.abs(y - sample).max() < 5e-3  # TODO: what is a sensible value?


def test_lookUp1D_linear(sampleData1D):
    N = 8192
    # prepare gpu
    table = theia.lookup.Table(sampleData1D[:, 1], interpolation="linear")
    gpu_table = table.upload()
    tensor = hp.FloatTensor(N)
    buffer = hp.FloatBuffer(N)
    code = compileShader("lookup.test.1D.glsl")
    program = hp.Program(code)
    program.bindParams(OutputBuffer=tensor)

    # run program
    class Push(Structure):
        _fields_ = [("table", c_int64), ("normalization", c_float)]

    push = Push(table=gpu_table.address, normalization=(N - 1.0))
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

    # check if LinearInterpolator produces same results
    interp = theia.lookup.createInterpolator(table)
    assert type(interp) is theia.lookup.LinearInterpolator
    y = interp(x)
    assert np.abs(y - buffer.numpy()).max() < 1e-6


def test_lookUp1D_cubic():
    N = 8192
    # unfortunately, the specific kind of cubic interpolation we use is neither
    # part of numpy nor scipy. Instead we will use a 2nd degree polynomial to
    # for samples. Cubic interpolation should be able to reconstruct it perfectly
    f = lambda x: 4 * x**2 - 1.2 * x + 10.0
    x = np.linspace(0.0, 1.0, 64)  # choose a small sample count, linear would fail
    y = f(x)

    # prepare GPU
    table = theia.lookup.Table(y, interpolation="cubic")
    gpu_table = table.upload()
    tensor = hp.FloatTensor(N)
    buffer = hp.FloatBuffer(N)
    program = hp.Program(compileShader("lookup.test.1D.glsl"))
    program.bindParams(OutputBuffer=tensor)

    # run program
    class Push(Structure):
        _fields_ = [("table", c_int64), ("normalization", c_float)]

    push = Push(table=gpu_table.address, normalization=(N - 1.0))
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(tensor, buffer))
        .Submit()
        .wait()
    )
    # recreate expected interpolation
    x = np.linspace(0, 1.0, N)
    y = f(x).astype(np.float32)
    # compare
    assert np.allclose(y, buffer.numpy())

    # check if CubicInterpolator produces the same results
    interp = theia.lookup.createInterpolator(table)
    assert type(interp) is theia.lookup.CubicInterpolator
    y = interp(x)
    assert np.allclose(y, buffer.numpy())


def test_lookUp1D_steffen():
    # again, this interpolation method is not part of scipy, but luckily it's
    # part of gvar. We will use values that provoke undulation with cubic
    # interpolation as a test case
    # fmt: off
    y_in = [0.16, 0.46, 0.944, 0.998, 0.999]
    y_out = [
       0.16      , 0.18837045, 0.21980437, 0.25430177, 0.29186264,
       0.33248699, 0.37617482, 0.42292612, 0.47321218, 0.53583028,
       0.6095551 , 0.6883542 , 0.76619516, 0.83704555, 0.89487295,
       0.93364493, 0.95073519, 0.96281991, 0.97307858, 0.98153697,
       0.98822087, 0.99315605, 0.9963683 , 0.99788339, 0.99818418,
       0.99840062, 0.99858377, 0.99873361, 0.99885016, 0.9989334 ,
       0.99898335, 0.999
    ]
    # fmt: on

    # prepare GPU
    table = theia.lookup.Table(y_in, interpolation="steffen")
    gpu_table = table.upload()
    N = len(y_out)
    tensor = hp.FloatTensor(N)
    buffer = hp.FloatBuffer(N)
    program = hp.Program(compileShader("lookup.test.1D.glsl"))
    program.bindParams(OutputBuffer=tensor)

    # run program
    class Push(Structure):
        _fields_ = [("table", c_int64), ("normalization", c_float)]

    push = Push(table=gpu_table.address, normalization=(N - 1.0))
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), N // 32))
        .Then(hp.retrieveTensor(tensor, buffer))
        .Submit()
        .wait()
    )

    # compare
    assert np.allclose(y_out, buffer.numpy())
    # check if CPU interpolator produces same results
    interp = theia.lookup.createInterpolator(table)
    assert type(interp) is theia.lookup.SteffenInterpolator
    y = interp(np.linspace(0, 1, len(y_out)))
    assert np.allclose(y, y_out)


@pytest.fixture
def sampleData2D():
    """Sample data for 2D lookup test. Shape: (x,y,f(x,y))"""
    x = np.linspace(0.0, 1.0, 50)
    y = np.linspace(0.0, 1.0, 100)
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
    bounds = (None, (3.0, 8.0))
    sample = theia.lookup.sampleTable2D(data, 100, 100, boundaries=bounds).samples
    sample = sample.flatten()
    assert sample.size == 100 * 100
    assert abs(sample[:100].min() - 3) < 1e-5
    assert abs(sample[:100].max() - 13.0) < 1e-5
    assert abs(sample[::100].min() - 3.0) < 1e-5
    assert abs(sample[::100].max() - 8.0) < 1e-5
    # test cubic spline
    sample = theia.lookup.sampleTable2D(sampleData2D, 250, 250, mode="cubic").samples
    sample = sample.flatten()
    x = np.linspace(0.0, 1.0, 250)
    y = np.linspace(0.0, 1.0, 250)
    x, y = np.meshgrid(x, y)
    z = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * (1.0 - y * y) * (1.0 - x * x)
    assert np.abs(z.flatten() - sample).max() < 5e-3  # TODO: Find a reasonable value


def test_lookUp2D_linear():
    N, Nx, Ny = 256, 32, 64
    # prepare gpu
    f = lambda x, y: 2.0 + np.cos(10.0 * x) * np.exp(y)
    table = theia.lookup.createTableFromFunction(f, Nx, Ny, interpolation="linear")
    gpu_table = table.upload()

    image = hp.Image(hp.ImageFormat.R32_SFLOAT, N, N)
    buffer = hp.FloatBuffer(N * N)
    program = hp.Program(compileShader("lookup.test.2D.glsl"))
    program.bindParams(outputImage=image)

    # run program
    class Push(Structure):
        _fields_ = [("table", c_int64), ("normalization", c_float)]

    push = Push(table=gpu_table.address, normalization=(N - 1.0))
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), N // 4, N // 4))
        .Then(hp.retrieveImage(image, buffer))
        .Submit()
        .wait()
    )

    # recreate expected linear interpolation
    x = np.linspace(0.0, 1.0, Nx)
    y = np.linspace(0.0, 1.0, Ny)
    z = f(*np.meshgrid(x, y, indexing="ij"))
    interp = RegularGridInterpolator((x, y), z)
    x = np.linspace(0.0, 1.0, N)
    x, y = np.meshgrid(x, x)
    p = np.stack((x.flatten(), y.flatten()), axis=-1)
    z = interp(p)
    assert np.abs(z - buffer.numpy()).max() < 5e-6

    # check LinearInterpolator produces the same values
    interp = theia.lookup.createInterpolator(table)
    assert type(interp) is theia.lookup.LinearInterpolator
    z = interp(p)
    assert np.abs(z - buffer.numpy()).max() < 5e-6


def test_lookUp2D_cubic():
    N, Nx, Ny = 512, 64, 48
    # again, no available Python implementation of the algo we use, so want to
    # interpolate a 2nd degree polynomial perfectly instead
    f = lambda x, y: 4.0 * x * x - 2.5 * y * y + 1.6 * x * y - 2.0 * x - y + 10.0
    table = theia.lookup.createTableFromFunction(f, Nx, Ny, interpolation="cubic")
    gpu_table = table.upload()
    image = hp.Image(hp.ImageFormat.R32_SFLOAT, N, N)
    buffer = hp.FloatBuffer(N * N)
    program = hp.Program(compileShader("lookup.test.2D.glsl"))
    program.bindParams(outputImage=image)

    # run program
    class Push(Structure):
        _fields_ = [("table", c_int64), ("normalization", c_float)]

    push = Push(table=gpu_table.address, normalization=(N - 1.0))
    (
        hp.beginSequence()
        .And(program.dispatchPush(bytes(push), N // 4, N // 4))
        .Then(hp.retrieveImage(image, buffer))
        .Submit()
        .wait()
    )

    # recreate expected linear interpolation
    x = np.linspace(0.0, 1.0, N)
    x, y = np.meshgrid(x, x)
    z = f(x.flatten(), y.flatten())
    assert np.allclose(z, buffer.numpy())

    # check if CubicInterpolator produces the same results
    interp = theia.lookup.createInterpolator(table)
    assert type(interp) is theia.lookup.CubicInterpolator
    p = np.stack([x.flatten(), y.flatten()], -1)
    zz = interp(p)
    assert np.allclose(zz, buffer.numpy())


def test_getTableSize(sampleData1D, sampleData2D):
    assert (
        theia.lookup.getTableSize(sampleData1D[:, 1])
        == theia.lookup.Table(sampleData1D[:, 1]).nbytes
    )
    assert (
        theia.lookup.getTableSize(sampleData2D[:, 2])
        == theia.lookup.Table(sampleData2D[:, 2]).nbytes
    )


def test_uploadTables(rng):
    t1 = rng.random((64,))
    t2 = rng.random((16,))
    t3 = rng.random((32,))

    tensor, ptr = theia.lookup.uploadTables([t1, t2, t3])

    tables = [
        theia.lookup.Table(t1),
        theia.lookup.Table(t2),
        theia.lookup.Table(t3),
    ]

    ptr_exp = tensor.address
    assert ptr_exp == ptr[0]
    ptr_exp += tables[0].nbytes
    assert ptr_exp == ptr[1]
    ptr_exp += tables[1].nbytes
    assert ptr_exp == ptr[2]

    size = sum(t.nbytes for t in tables)
    buffer = hp.ByteBuffer(size)
    hp.execute(hp.retrieveTensor(tensor, buffer))

    expected = np.empty(size, dtype=np.uint8)
    ptr = expected.ctypes.data
    for t in tables:
        ptr += t.copy(ptr)
    assert np.equal(buffer.numpy(), expected).all()
