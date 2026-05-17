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
    *axis_range: tuple[float, float]
        Optional ranges for each dimension or axis. If not specified, assumes
        [0,1] for each.
    interpolation: Interpolation, default="linear"
        Interpolation method to apply

    Note
    ----
    Data is stored as 32-bit floats.
    """

    ALIGNMENT: Final[int] = 32
    """Memory alignment requirement on the GPU"""
    PADDING: Final[int] = 1
    """Amount of entries padded at both ends of each axis"""

    def __init__(
        self,
        data: ArrayLike,
        *axis_range: tuple[float, float],
        interpolation: Interpolation = "linear",
    ) -> None:
        self._data = np.ascontiguousarray(_padTable(data), dtype=np.float32)
        self._shape = np.array(self._data.shape, dtype=np.int32)
        # subtract padding
        self._shape -= 2 * Table.PADDING
        self._interpolation = _toMethod(interpolation)
        # check ranges
        if len(axis_range) == 0:
            self._range = ((0.0, 1.0),) * self._data.ndim
        elif len(axis_range) != self._data.ndim:
            raise ValueError("Number of specified axis ranges must match data")
        else:
            self._range = axis_range
        # ideally Table wouldn't need to know this, but for now we do the test here
        if self.interpolation is InterpolationMethod.STEFFEN and self._data.ndim != 1:
            raise ValueError("Steffen interpolation is only defined for 1D data!")

    @staticmethod
    def createConstantValuePtr(value: float) -> int:
        """Creates a table pointer containing a constant value instead."""
        # Since we require 32 byte alignment, a valid pointer will always have its
        # lowest 5 bits set to zero. We can therefore use the lowest bit as flag
        # to mark constant values and store the value in the upper word
        # This way we can look up constant values directly from a property table
        return unpack("<Q", pack("<If", 0xF, value))[0]

    @property
    def axisRanges(self) -> tuple[tuple[float, float], ...]:
        """Ranges for data axis"""
        return self._range

    @property
    def constant(self) -> bool:
        """Whether the table is constant, i.e. only contains a single sample"""
        return np.all(self.shape == 1).item()

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
        # header consists of:
        #  - an integer for interpolation method
        #  - 2 floats per dimension for range
        #  - 1 int per dimension for sample count
        return 12 * self.ndim + 4 + self.data.nbytes

    @property
    def ndim(self) -> int:
        """Number of dimension in this table"""
        return len(self.shape)

    @property
    def samples(self) -> NDArray[np.float32]:
        """The actual sampled data, i.e. `data` without the padding."""
        mask = [slice(Table.PADDING, -Table.PADDING)] * self.ndim
        return self.data[*mask]

    def copy(self, ptr, *, unsafe: bool = False) -> int:
        """
        Copies the table to the given memory address and returns the amount of
        copied bytes.
        """
        # check alignment
        if (ptr & (Table.ALIGNMENT - 1)) != 0 and not unsafe:
            raise ValueError("memory address is not properly aligned!")

        # write header
        header = self._packHeader()
        memmove(ptr, header, len(header))
        ptr = ptr + len(header)
        # copy data
        memmove(ptr, self._data.ctypes.data, self._data.nbytes)
        return self.nbytes

    def upload(self) -> hp.Tensor:
        """Uploads the table to the GPU"""
        buffer = hp.Buffer(self.nbytes)
        tensor = hp.Tensor(self.nbytes, alignment=Table.ALIGNMENT)
        self.copy(buffer.address)
        hp.execute(hp.updateTensor(buffer, tensor))
        return tensor

    def save(self, file: BufferedIOBase) -> None:
        """Writes the table to the given file."""
        np.savez(
            file,
            samples=self.samples,
            interpolation=int(self.interpolation),
            ranges=np.array(self.axisRanges),
        )

    @classmethod
    def load(cls, file: BufferedIOBase) -> Self | None:
        """
        Loads the next table stored in the given file. Returns `None` if end
        of file has been reached.
        """
        try:
            data = np.load(file)
        except EOFError:
            return None
        # extract meta data
        axisRanges = tuple(map(tuple, data["ranges"].tolist()))
        interpolation: int = data["interpolation"].item()
        if interpolation not in InterpolationMethod:
            raise RuntimeError(f"Unknown interpolation method {interpolation}")
        interpolation = InterpolationMethod(interpolation)
        # create table
        return cls(data["samples"], *axisRanges, interpolation=interpolation)

    def _packHeader(self) -> bytes:
        """util function assembling the header"""
        result = b""
        for axis in self.axisRanges:
            result += pack("<ff", *axis)
        result += pack("<i", self.interpolation)
        result += self.shape.tobytes()
        return result


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

    result = 0  # words
    # header
    result += 2 * len(a)  # ranges
    result += 1  # interpolation
    result += len(a)  # sample count per axis
    # padding
    padded = tuple(axis + 2 * Table.PADDING for axis in a)
    result += np.prod(padded).item()
    # return size
    return result * 4


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
    size = 0
    align_up = lambda x, a: (x + a - 1) & -a
    for table in tables:
        size = align_up(size, Table.ALIGNMENT)
        size += table.nbytes

    buffer = hp.Buffer(size)
    tensor = hp.Tensor(size, alignment=Table.ALIGNMENT)

    adr_list = []
    adr = tensor.address
    ptr = buffer.address
    align_pad = lambda x, a: (a - (x % a)) % a  # (adr, alignment) -> padding
    for table in tables:
        pad = align_pad(adr, Table.ALIGNMENT)
        adr += pad
        ptr += pad
        adr_list.append(adr)
        n = table.copy(ptr)
        adr += n
        ptr += n

    # sanity check
    assert adr == tensor.address + size
    assert ptr == buffer.address + size

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
    ranges = tuple((a[0].item(), a[-1].item()) for a in axes)
    grid = np.meshgrid(*axes, indexing="ij")
    values = f(*grid)
    return Table(values, *ranges, interpolation=interpolation)


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
    _range = (x[0], x[-1])
    if mode == "linear":
        return Table(np.interp(x, data[:, 0], data[:, 1]), _range)
    elif mode == "cubic":
        spline = CubicSpline(data[:, 0], data[:, 1])
        return Table(spline(x), _range)
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
    if boundaries == None:
        x = _parseBoundary(data[:, 0], None, nx)
        y = _parseBoundary(data[:, 1], None, ny)
    elif type(boundaries) == tuple:
        if len(boundaries) != 2:
            raise RuntimeError("Cant parse given boundaries!")
        x = _parseBoundary(data[:, 0], boundaries[0], nx)
        y = _parseBoundary(data[:, 1], boundaries[1], ny)
    else:
        raise ValueError("Unexpected boundary format")
    x_range = (x[0], x[-1])
    y_range = (y[0], y[-1])
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
    return Table(values, x_range, y_range)


class Interpolator(ABC):
    """
    Base class for algorithms interpolating values stored in a `Table`.

    Parameter
    ---------
    table: Table
        Table containing the values to be interpolated
    """

    def __init__(
        self,
        table: Table,
    ) -> None:
        self._table = table

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
        for i, (a, b) in enumerate(self.table.axisRanges):
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
    """

    def __init__(
        self,
        table: Table,
    ) -> None:
        super().__init__(table)

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
    """

    def __init__(
        self,
        table: Table,
    ) -> None:
        super().__init__(table)

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
    """

    def __init__(
        self,
        table: Table,
    ) -> None:
        if table.ndim != 1:
            raise ValueError("Steffen interpolation is only defined for 1D data!")
        super().__init__(table)
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
        assumes (0, 1) for every dimension. Ignored if `data` is already a `Table`.
    """
    if bounds is None:
        bounds = tuple()
    table = data if isinstance(data, Table) else Table(data, *bounds)
    if method is None:
        method = table.interpolation
    else:
        method = _toMethod(method)

    if method in _interpolatorDict:
        return _interpolatorDict[method](table)
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
