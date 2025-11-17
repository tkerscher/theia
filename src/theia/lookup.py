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
from struct import pack, unpack

from abc import ABC, abstractmethod
from enum import IntEnum
from io import BufferedIOBase
from typing import Final, Literal, Type
from typing_extensions import Self  # introduced in 3.11

__all__ = [
    "createInterpolator",
    "createTableFromFunction",
    "getTableSize",
    "interpolate",
    "sampleTable1D",
    "sampleTable2D",
    "uploadTables",
    "CubicInterpolator",
    "InterpolationMethod",
    "Interpolator",
    "LinearInterpolator",
    "SteffenInterpolator",
    "Table",
]


def __dir__():
    return __all__


class InterpolationMethod(IntEnum):
    """Enumeration of available interpolation methods"""

    LINEAR = 1
    CUBIC = 2
    STEFFEN = 3


Interpolation = InterpolationMethod | Literal["linear", "cubic", "steffen"]


def _toMethod(i: Interpolation) -> InterpolationMethod:
    """util function converting Interpolation to InterpolationMethod"""
    if type(i) is InterpolationMethod:
        return i
    else:
        return InterpolationMethod[str(i).upper()]


def _padTable(data: ArrayLike) -> NDArray:
    """
    Pads the given table with the appropriate values dictated by the boundary
    conditions of cubic convolution interpolation.
    """
    # allocate array
    result = np.pad(data, 1)
    # calculate padding for each dimension
    # view gives mask to select idx at given axis
    view = lambda axis, idx: (*[slice(None)] * axis, idx, ...)
    for i, s in enumerate(result.shape):
        if s == 3:
            # constant extrapolate
            result[*view(i, (0, 2))] = result[*view(i, 1)]
        elif s == 4:
            # linear extrapolate
            delta = result[*view(i, 1)] - result[*view(i, 2)]
            result[*view(i, 0)] = result[*view(i, 1)] - delta
            result[*view(i, -1)] = result[*view(i, -1)] + delta
        elif s > 4:
            # cubic extrapolate
            result[*view(i, 0)] = (
                3.0 * result[*view(i, 1)]
                - 3.0 * result[*view(i, 2)]
                + result[*view(i, 3)]
            )
            result[*view(i, -1)] = (
                3.0 * result[*view(i, -2)]
                - 3.0 * result[*view(i, -3)]
                + result[*view(i, -4)]
            )
    # done
    return result


class Table:
    """
    Encapsulates data used in look up tables on the GPU. Provides methods to
    create and upload them to the GPU. To minimize the amount of required memory
    fetches during interpolation, expects the data as the values at equidistant
    sampling points.

    Memory layout of the data follows the C style, i.e. it is contiguous in the
    last axis.

    Parameters
    ----------
    data: ArrayLike
        Values of the equidistant sampling points used for interpolation.
        Will be converted to floats.
    interpolation: Interpolation, default="linear"
        Interpolation method to apply

    Note
    ----
    Data is stored as 32-bit floats.
    """

    ALIGNMENT: Final[int] = 4
    """Memory alignment requirement on the GPU"""
    PADDING: Final[int] = 1
    """Amount of entries padded at both ends of each axis"""

    def __init__(
        self,
        data: ArrayLike,
        *,
        interpolation: Interpolation = "linear",
    ) -> None:
        self._data = np.ascontiguousarray(_padTable(data), dtype=np.float32)
        self._shape = np.array(self._data.shape, dtype=np.int32)
        # subtract padding
        self._shape -= 2 * Table.PADDING
        self._interpolation = _toMethod(interpolation)
        # ideally Table wouldn't need to know this, but for now we do the test here
        if self.interpolation is InterpolationMethod.STEFFEN and self._data.ndim != 1:
            raise ValueError("Steffen interpolation is only defined for 1D data!")

    @property
    def data(self) -> NDArray[np.float32]:
        """The underlying data"""
        return self._data

    @property
    def interpolation(self) -> InterpolationMethod:
        """Method used to interpolate values"""
        return self._interpolation

    @property
    def shape(self) -> NDArray[np.int32]:
        """Header of the table containing sample sizes for each dimension"""
        return self._shape

    @property
    def nbytes(self) -> int:
        """Required amount of bytes to store the table"""
        return self._data.nbytes + self._shape.nbytes + 4

    @property
    def ndim(self) -> int:
        """Number of dimension in this table"""
        return len(self.shape)

    @property
    def samples(self) -> NDArray[np.float32]:
        """The actual sampled data, i.e. `data` without the padding."""
        mask = [slice(Table.PADDING, -Table.PADDING)] * self.ndim
        return self.data[*mask]

    def copy(self, ptr) -> int:
        """
        Copies the table to the given memory address and returns the amount of
        copied bytes.
        """
        header = np.append(np.int32(self.interpolation), self.shape)
        memmove(ptr, header.ctypes.data, header.nbytes)
        ptr = ptr + header.nbytes  # dont alter initial ptr
        memmove(ptr, self._data.ctypes.data, self._data.nbytes)
        return self.nbytes

    def upload(self) -> hp.Tensor:
        """Uploads the table to the GPU"""
        buffer = hp.Buffer(self.nbytes)
        tensor = hp.Tensor(self.nbytes)
        self.copy(buffer.address)
        hp.execute(hp.updateTensor(buffer, tensor))
        return tensor

    def save(self, file: BufferedIOBase) -> None:
        """
        Writes the table to the given file.
        Returns the total amount of bytes written.
        """
        # first, write the header. currently this only
        # consists of the interpolation method
        file.write(pack("<i", self.interpolation))
        # followed by the actual numpy data
        np.save(file, self.samples)

    @classmethod
    def load(cls, file: BufferedIOBase) -> Self | None:
        """
        Loads the next table stored in the given file. Returns `None` if end
        of file has been reached.
        """
        # first read interpolation method. will be empty if we already reached eof
        # TODO: we are not guaranteed to get 4 bytes. E.g. on TCP streams, we might
        #       get less, requiring additional read commands. For now this should
        #       be fine though
        header = file.read(4)
        if not header or len(header) < 4:
            return None
        interpolation = unpack("<i", header)[0]
        if interpolation not in InterpolationMethod:
            raise RuntimeError(f"Unknown interpolation method {interpolation}")
        interpolation = InterpolationMethod(interpolation)
        samples = np.load(file)
        return cls(samples, interpolation=interpolation)


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
    # header + padding + data
    return (1 + len(a) + 2 * Table.PADDING * len(a) + sum(a)) * 4  # 4 bytes per item


def uploadTables(data: list[NDArray]) -> tuple[hp.Tensor, list[int]]:
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
    tensor: Tensor
        Tensor containing the table data
    addresses: list[int]
        Device addresses pointing to the individual tables on the device.
    """
    tables = [Table(d) for d in data]
    size = sum([table.nbytes for table in tables])

    buffer = hp.Buffer(size)
    tensor = hp.Tensor(size)

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


def createTableFromFunction(
    f,
    *xi,
    interpolation: Interpolation = "linear",
) -> Table:
    """
    Creates a table by sampling the given function on a regular grid with the
    given dimension.

    Parameters
    ----------
    f: Callable (x1,x2,...,xn) -> float
        Function to be repeatedly evaluated to create data points
    x1,x2,...,xn: int | (float, float, int)
        Specify the number of samples per dimension and optionally the boundary
        for each axis in the form (min, max, n_samples)
    interpolation: Interpolation, default="linear"
        Interpolation method of the produced table

    Returns
    -------
        Table in suitable format for GPU interpolation.
    """

    def createAxis(spec: int | tuple[float, float, int]) -> NDArray:
        if type(spec) == int:
            return np.linspace(0.0, 1.0, spec)
        elif type(spec) == tuple and len(spec) == 3:
            return np.linspace(*spec)
        else:
            raise ValueError(f"Cannot parse dimension spec: {spec}")

    axes = [createAxis(x) for x in xi]
    grid = np.meshgrid(*axes, indexing="ij")
    values = f(*grid)
    return Table(values, interpolation=interpolation)


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


class Interpolator(ABC):
    """
    Base class for algorithms interpolating values stored in a `Table`.

    Parameter
    ---------
    table: Table
        Table containing the values to be interpolated
    bounds: tuple[(float, float), ...] | None, default=None
        Boundaries of each dimension specified as (min, max). If `None`,
        assumes (0, 1) for every dimension.
    """

    def __init__(
        self,
        table: Table,
        *,
        bounds: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        self._table = table
        self._bounds = bounds

    @property
    def bounds(self) -> tuple[tuple[float, float], ...] | None:
        """
        Boundaries of each dimesion used for interpolation.
        If `None`, assumes (0,1) for every dimensions.
        """
        return self._bounds

    @property
    def table(self) -> Table:
        """Underlying table containing values for interpolation"""
        return self._table

    def _toGridCoords(self, xi: ArrayLike) -> tuple[NDArray, NDArray]:
        """
        Converts the given coordinates to grid coordinates using the specified
        bounds. Evaluates for each coordinate x_i the range x_k <= x_i <= x_k+1
        returning both k and u = (x_i - x_k) / (x_k+1 - x_k)
        """
        # we demote to 32bit float to mimic the GPU more closely
        xi = np.asarray(xi, dtype=np.float32, copy=True)
        if xi.ndim == 1:
            xi = xi.reshape((-1, 1))
        elif xi.ndim >= 3 or xi.shape[1] != self.table.ndim:
            raise ValueError(f"xi has wrong shape! Expected (...,{self.table.ndim})!")
        # convert xi -> (0,1)
        if self.bounds is not None:
            for i, (a, b) in enumerate(self.bounds):
                xi[:, i] = (xi[:, i] - a) / (b - a)
        # convert to grid coords
        np.clip(xi, 0.0, 1.0, out=xi)
        for i, n in enumerate(self.table.shape):
            xi[:, i] *= n - 1
        # split in integer and fractional part
        k = np.floor(xi).astype(np.int32)
        for i, n in enumerate(self.table.shape):
            # to prevent out of bounds access, ensure k is not the last element
            np.clip(k[:, i], 0, n - 2, out=k[:, i])
        u = xi - k
        # add padding
        k += Table.PADDING
        return k, u

    @abstractmethod
    def __call__(self, xi: ArrayLike) -> NDArray:
        """
        Interpolates the underlying table at the given positions.

        Parameters
        ----------
        xi: NDArray of shape (..., ndim)
            Positions to evaluate the interpolation at
        """
        ...


def _step(x: NDArray, axis: int, delta: int) -> NDArray:
    """util function adding `delta` along axis"""
    x = x.copy()
    x[:, axis] += delta
    return x


class LinearInterpolator(Interpolator):
    """
    Linear interpolation of `Table`.

    Parameter
    ---------
    table: Table
        Table containing the values to be interpolated
    bounds: tuple[(float, float), ...] | None, default=None
        Boundaries of each dimension specified as (min, max). If `None`,
        assumes (0, 1) for every dimension.
    """

    def __init__(
        self,
        table: Table,
        *,
        bounds: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        super().__init__(table, bounds=bounds)

    def _interp(self, k: NDArray, u: NDArray, axis: int) -> NDArray:
        if axis >= self.table.ndim:
            return self.table.data[*k.T]
        else:
            a = self._interp(k, u, axis + 1)
            b = self._interp(_step(k, axis, 1), u, axis + 1)
            return a * (1.0 - u[:, axis]) + b * u[:, axis]

    def __call__(self, xi: ArrayLike) -> NDArray:
        k, u = self._toGridCoords(xi)
        return self._interp(k, u, 0)


class CubicInterpolator(Interpolator):
    """
    Cubic interpolation based on the unique solution presented in:
    R.G. Keys: "Cubic Convolution Interpolation for Digital Image Processing" (1981)

    Parameter
    ---------
    table: Table
        Table containing the values to be interpolated
    bounds: tuple[(float, float), ...] | None, default=None
        Boundaries of each dimension specified as (min, max). If `None`,
        assumes (0, 1) for every dimension.
    """

    def __init__(
        self,
        table: Table,
        *,
        bounds: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        super().__init__(table, bounds=bounds)

    def _interp(self, k: NDArray, u: NDArray, axis: int) -> NDArray:
        if axis >= self.table.ndim:
            return self.table.data[*k.T]
        else:
            a = self._interp(_step(k, axis, -1), u, axis + 1)
            b = self._interp(k, u, axis + 1)
            c = self._interp(_step(k, axis, 1), u, axis + 1)
            d = self._interp(_step(k, axis, 2), u, axis + 1)

            s = u[:, axis]
            sa = -(s**3) + 2.0 * s**2 - s
            sb = 3.0 * s**3 - 5.0 * s**2 + 2.0
            sc = -3.0 * s**3 + 4.0 * s**2 + s
            sd = s**3 - s**2

            return 0.5 * (sa * a + sb * b + sc * c + sd * d)

    def __call__(self, xi: ArrayLike) -> NDArray:
        k, u = self._toGridCoords(xi)
        return self._interp(k, u, 0)


class SteffenInterpolator(Interpolator):
    """
    Local monotone cubic hermite interpolation in one dimension based on the
    paper by M. Steffen:
    M. Steffen: A simple method for monotonic inerpolation in one dimension (1990)

    Parameters
    ----------
    table: Table
        Table containing the values to be interpolated
    bounds: tuple[(float, float), ...] | None, default=None
        Boundaries of each dimension specified as (min, max). If `None`,
        assumes (0, 1) for every dimension.
    """

    def __init__(
        self,
        table: Table,
        *,
        bounds: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        if table.ndim != 1:
            raise ValueError("Steffen interpolation is only defined for 1D data!")
        super().__init__(table, bounds=bounds)
        # pre calc slopes to declutter notation later on
        y = table.data
        s = np.diff(table.data)
        dy = np.stack([np.abs(s[:-1]), np.abs(s[1:]), 0.25 * np.abs(y[2:] - y[:-2])])
        dy = (np.sign(s[1:]) + np.sign(s[:-1])) * np.min(dy, 0)
        self._s, self._dy = s, dy

    def __call__(self, xi: ArrayLike) -> NDArray:
        k, u = self._toGridCoords(xi)
        k, u = k.squeeze(), u.squeeze()

        y1 = self._table.data[k]
        s1 = self._s[k]
        dy1 = self._dy[k - 1]
        dy2 = self._dy[k]

        a = dy1 + dy2 - 2.0 * s1
        b = 3.0 * s1 - 2.0 * dy1 - dy2
        return ((a * u + b) * u + dy1) * u + y1


_interpolatorDict: dict[InterpolationMethod, Type[Interpolator]] = {
    InterpolationMethod.LINEAR: LinearInterpolator,
    InterpolationMethod.CUBIC: CubicInterpolator,
    InterpolationMethod.STEFFEN: SteffenInterpolator,
}


def createInterpolator(
    data: ArrayLike | Table,
    *,
    method: Interpolation | None = None,
    bounds: tuple[tuple[float, float], ...] | None = None,
) -> Interpolator:
    """
    Creates an interpolator of given method using the specified data.

    Parameters
    ----------
    data: ArrayLike | Table
        Sampled data used to build the underlying table
    method: Interpolation | None, default=None
        Interpolation method to use. If `None`, uses the one defined by the
        table.
    bounds: tuple[(float, float), ...] | None, default=None
        Boundaries of each dimension specified as (min, max). If `None`,
        assumes (0, 1) for every dimension.
    """
    table = data if isinstance(data, Table) else Table(data)
    if method is None:
        method = table.interpolation
    else:
        method = _toMethod(method)

    if method in _interpolatorDict:
        return _interpolatorDict[method](table, bounds=bounds)
    else:
        raise ValueError("Unknown interpolation method!")


def interpolate(
    data: ArrayLike | Table,
    xi: ArrayLike,
    *,
    method: Interpolation | None = None,
    bounds: tuple[tuple[float, float], ...] | None = None,
) -> NDArray:
    """
    Interpolates the given data using the specified method. Thin wrapper
    function around the corresponding interpolation classes.

    Parameters
    ----------
    data: ArrayLike | Table
        Sampled data used to build the underlying table
    xi: ArrayLike
        Coordinates at which to evaluate the interpolation
    method: Interpolation | None, default=None
        Interpolation method to use. If `None`, uses the one defined by the
        table.
    bounds: tuple[(float, float), ...] | None, default=None
        Boundaries of each dimension specified as (min, max). If `None`,
        assumes (0, 1) for every dimension.
    """
    return createInterpolator(data, method=method, bounds=bounds)(xi)
