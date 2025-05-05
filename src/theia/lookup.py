from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    CubicSpline,
    LinearNDInterpolator,
)

import hephaistos as hp
from ctypes import memmove
from hephaistos import ByteTensor

from typing import Final, Literal

__all__ = [
    "evalTable",
    "getTableSize",
    "sampleTable1D",
    "sampleTable2D",
    "uploadTables",
    "Table",
]


def __dir__():
    return __all__


class Table:
    """
    Encapsulates data used in look up tables on the GPU. Provides methods to
    create and upload them to the GPU. To minimize the amount of required memory
    fetches during interpolation, expects the data as the values at equidistant
    sampling points.

    Parameters
    ----------
    data: ArrayLike
        Values of the equidistant sampling points used for interpolation.
        Will be converted to floats.

    Note
    ----
    Data is stored as 32-bit floats.
    """

    ALIGNMENT: Final[int] = 4
    """Memory alignment requirement on the GPU"""

    def __init__(self, data: ArrayLike) -> None:
        self._data = np.ascontiguousarray(data, dtype=np.float32)
        self._header = np.array(self._data.shape, dtype=np.int32)

    @property
    def data(self) -> NDArray[np.float32]:
        """The underlying data"""
        return self._data

    @property
    def nbytes(self) -> int:
        """Required amount of bytes to store the table"""
        return self._data.nbytes + self._header.nbytes

    def copy(self, ptr) -> int:
        """
        Copies the table to the given memory address and returns the amount of
        copied bytes.
        """
        memmove(ptr, self._header.ctypes.data, self._header.nbytes)
        ptr = ptr + self._header.nbytes  # dont alter initial ptr
        memmove(ptr, self._data.ctypes.data, self._data.nbytes)
        return self.nbytes

    def upload(self) -> hp.ByteTensor:
        """Uploads the table to the GPU"""
        buffer = hp.RawBuffer(self.nbytes)
        tensor = hp.ByteTensor(self.nbytes)
        self.copy(buffer.address)
        hp.execute(hp.updateTensor(buffer, tensor))
        return tensor


def getTableSize(a: ArrayLike | tuple[int, ...] | None) -> int:
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


def uploadTables(data: list[NDArray]) -> tuple[ByteTensor, list[int]]:
    """
    Creates a table for each data entry in the given list and uploads them to
    the GPU. Returns the tensor storing them and a the corresponding list of
    device addresses pointing to the uploaded tables.

    Parameters
    ----------
    data: list[NDArray]
        List of data used to populate tables.

    Returns
    -------
    tensor: ByteTensor
        Tensor containing the table data
    addresses: list[int]
        Device addresses pointing to the individual tables on the device.
    """
    tables = [Table(d) for d in data]
    size = sum([table.nbytes for table in tables])

    buffer = hp.RawBuffer(size)
    tensor = ByteTensor(size)

    adr_list = []
    adr = tensor.address
    ptr = buffer.address
    for table in tables:
        adr_list.append(adr)
        n = table.copy(ptr)
        adr += n
        ptr += n

    hp.execute(hp.updateTensor(buffer, tensor))

    return tensor, adr_list


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
) -> Table:
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
        Table in suitable format for GPU interpolation
    """
    x = _parseBoundary(data[:, 0], boundary, nx)
    if mode == "linear":
        return Table(np.interp(x, data[:, 0], data[:, 1]))
    elif mode == "cubic":
        spline = CubicSpline(data[:, 0], data[:, 1])
        return Table(spline(x))
    else:
        raise RuntimeError("Unknown interpolation mode!")


def sampleTable2D(
    data,
    nx=1024,
    ny=1024,
    *,
    boundaries=None,
    mode: Literal["linear", "cubic"] = "linear",
) -> Table:
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
    boundaries: None or (min, max) for both or each dimension (tuple of tuple), default = None
        boundaries of the interpolated data. Inferred from data if None.
        Can be specified for both dimensions or each individually
    mode: "linear" | "cubic"
        type of interpolation

    Returns
    -------
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
    return Table(values)


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
    return Table(data)
