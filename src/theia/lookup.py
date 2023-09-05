import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    CubicSpline,
    LinearNDInterpolator,
)
from typing import Literal, Union


def getTableSize(a: Union[ArrayLike, tuple[int, ...], None]) -> int:
    """
    Calculates the size in bytes needed to store a table of given shape on the
    GPU. Returns zero if a is None.
    """
    if a is None:
        return 0
    if type(a) != tuple:
        a = np.shape(a)
    if len(a) == 0:
        raise RuntimeError("table cannot have zero shape!")
    # header + data
    return (len(a) + sum(a)) * 4  # 4 bytes per float


def createTable(data):
    """
    Transform the given data to be suitable for table lookups on the GPU.
    It is assumed that the data is uniformly sampled on [0,1] on each dimension.

    Parameters
    ----------
    data : ndarray
        Data to be converted in table format

    Returns
    -------
    ndarray
        Table in suitable format for GPU interpolation
    """
    if data.ndim > 2:
        raise RuntimeError("data must be one or two dimensional!")

    # header is a list with the size of each dimension minus one
    header = np.array(data.shape) - 1.0
    data = np.concatenate([header, data.flatten()])
    return np.ascontiguousarray(data, np.float32)


def _parseBoundary(data, boundary, n):
    """Helper function to create grid points from data and given boundaries"""
    if boundary == None:
        return np.linspace(data.min(), data.max(), n)
    elif type(boundary) == tuple and len(boundary) == 2:
        return np.linspace(boundary[0], boundary[1], n)
    else:
        raise RuntimeError("Cant parse given boundaries!")


def sampleTable1D(
    data, nx=1024, *, boundary=None, mode: Literal["linear", "cubic"] = "linear"
) -> NDArray[np.float32]:
    """
    Creates a 1D table by interpolating the given data either linearly or with
    cubic splines. The data must be provided as a ndarray of shape (N,2) and
    form (x,f(x)).
    The returned array has the right format to be uploaded to the GPU.

    Parameters
    ----------
    data: ndarray of shape(N,2)
        data in grid form (x,f(x)) to be sampled from
    nx: int, default = 1024
        number of regular placed sample points
    boundary: None or (min, max), default = None
        boundaries of the interpolated data. Inferred from data if None
    mode: "linear" | "cubic"
        type of interpolation

    Returns
    -------
    ndarray
        Table in suitable format for GPU interpolation
    """
    x = _parseBoundary(data[:, 0], boundary, nx)
    if mode == "linear":
        return createTable(np.interp(x, data[:, 0], data[:, 1]))
    elif mode == "cubic":
        spline = CubicSpline(data[:, 0], data[:, 1])
        return createTable(spline(x))
    else:
        raise RuntimeError("Unknown interpolation mode!")


def sampleTable2D(
    data,
    nx=1024,
    ny=1024,
    *,
    boundaries=None,
    mode: Literal["linear", "cubic"] = "linear",
) -> NDArray[np.float32]:
    """
    Creates a 2D table by interpolating the given data either linearly or with
    cubic splines. The data must be provided as a ndarray of shape (N,2) and
    form (x,y,f(x,y)).
    The returned array has the right format to be uploaded to the GPU.

    Parameters
    ----------
    data: ndarray of shape(N,3)
        data in grid form (x,y,f(x,y)) to be sampled from
    nx: int, default = 1024
        number of regular placed sample points in x direction
    ny: int, default = 1024
        number of regular placed sample points in y direction
    boundary: None or (min, max) for both or each dimension (tuple of tuple), default = None
        boundaries of the interpolated data. Inferred from data if None.
        Can be specified for both dimensions or each individually
    mode: "linear" | "cubic"
        type of interpolation

    Returns
    -------
    ndarray
        Table in suitable format for GPU interpolation
    """
    # parse boundaries
    x = y = None
    if boundaries == None:
        x = _parseBoundary(data[:, 0], None, nx)
        y = _parseBoundary(data[:, 1], None, ny)
    elif type(boundaries) == tuple:
        if len(boundaries) != 2:
            raise RuntimeError("Cant parse given boundaries!")
        x = _parseBoundary(data[:, 0], boundaries[0], nx)
        y = _parseBoundary(data[:, 1], boundaries[1], ny)
    # create mesh grid
    x, y = np.meshgrid(x, y)

    # set interpolation mode
    model = None
    if mode == "linear":
        model = LinearNDInterpolator
    elif mode == "cubic":
        model = CloughTocher2DInterpolator
    else:
        raise RuntimeError("Unknown interpolation mode!")

    # interpolate
    interp = model(data[:, :2], data[:, 2])
    values = interp(x, y)
    return createTable(values)


def evalTable(f, *ai):
    """
    Creates a table by sampling the given function on a regular grid.
    It can process both one and two dimensional functions.
    The result has the correct format to be uploaded to the GPU.

    Parameters
    ----------
    f: Callable accepting as many floats as specified dims
        Function to be repeatedly called to create data points
    a1,a2,...,an: n_samples or (min, max, n_samples)
        Specify the number of samples per dimension and optionally the boundary
        for each axis

    Returns
    -------
        Table in suitable format for GPU interpolation
    """

    # create sample points
    def createAxis(spec):
        if type(spec) == int:
            return np.linspace(0.0, 1.0, spec)
        elif type(spec) == tuple and len(spec) == 3:
            return np.linspace(*spec)
        raise RuntimeError(f"Cannot parse dimension: {spec}")

    axes = [createAxis(a) for a in ai]
    grid = np.meshgrid(*axes)

    # sample function
    values = f(*grid)
    axes.push(values)
    data = np.stack(axes, axis=-1)

    # create table and return
    return createTable(data)
