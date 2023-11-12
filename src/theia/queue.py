from __future__ import annotations
import hephaistos as hp

from ctypes import Structure, c_int32, sizeof
from numpy import ndarray
from numpy.ctypeslib import as_array

from numpy.typing import NDArray
from typing import Any, Optional, Set, Type, Union


class QueueView:
    """
    View allowing structured access to a queue stored in memory

    Parameters
    ----------
    data: int
        address of the memory the view should point at
    item: Structure
        Structure describing a single item
    capacity: int
        Maximum number of items the queue can hold
    skipHeader: bool, default=False
        True, if data does not contain a queue header
    """

    class Header(Structure):
        _fields_ = [("count", c_int32)]

    def __init__(
        self,
        data: int,
        item: Type[Structure],
        capacity: int,
        *,
        skipHeader: bool = False,
    ) -> None:
        # store item type
        self._item = item
        self._names = {name for name, t in item._fields_}
        self._capacity = capacity
        # read header
        if not skipHeader:
            self._header = self.Header.from_address(data)
            data += sizeof(self.Header)
        else:
            self._header = None

        # # create SoA and store it
        # class SoA(Structure):
        #     _fields_ = [(name, t * capacity) for name, t in item._fields_]
        # self._data = SoA.from_address(data)

        self._data = (item * capacity).from_address(data)
        self._arr = as_array(self._data)

    def __len__(self) -> int:
        return self._capacity

    def __contains__(self, key: str) -> bool:
        return key in self._names

    def __getitem__(self, key) -> Union[QueueView, NDArray]:
        if isinstance(key, slice) or isinstance(key, tuple) or isinstance(key, ndarray):
            return QueueSubView(self, key)
        if key not in self:
            raise KeyError(f"No field with name {key}")
        # return as_array(getattr(self._data, key))
        return self._arr[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self:
            raise KeyError(f"No field with name {key}")
        # as_array(getattr(self._data, key))[:] = value
        self._arr[key][:] = value

    @property
    def capacity(self) -> int:
        """Capacity of the queue"""
        return self._capacity

    @property
    def fields(self) -> Set[str]:
        """List of field names"""
        return self._names

    @property
    def hasHeader(self) -> bool:
        """True if the queue has a header"""
        return self._header is not None

    @property
    def count(self) -> int:
        """Number of items in queue. Equal to capacity if header was skipped"""
        if self._header is not None:
            return self._header.count
        else:
            return self.capacity

    @count.setter
    def count(self, value: int) -> None:
        if self._header is not None:
            self._header.count = value
        else:
            raise ValueError("The queue has no header and this no count can be set!")

    @property
    def item(self) -> Type[Structure]:
        """Structure describing the items of the queue"""
        return self._item


class QueueSubView:
    """Utility class for slicing and masking a QueueView"""

    def __init__(
        self, orig: Union[QueueView, QueueSubView], mask: Union[slice, tuple, NDArray]
    ) -> None:
        self._orig = orig
        self._mask = mask
        # query new length by applying mask to a random field
        self._count = len(orig[next(iter(orig.fields))][mask])

    def __len__(self) -> int:
        return self._count

    def __contains__(self, key: str) -> bool:
        return key in self._orig

    def __getitem__(self, key) -> Union[QueueView, NDArray]:
        if isinstance(key, slice) or isinstance(key, tuple) or isinstance(key, ndarray):
            return QueueSubView(self, key)
        if key not in self:
            raise KeyError(f"No field with name {key}")
        return self._orig[key][self._mask]

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self:
            raise KeyError(f"No field with name {key}")
        self._orig[key][self._mask] = value

    @property
    def fields(self) -> Set[str]:
        """List of field names"""
        return self._orig.fields


def queueSize(
    item: Union[Type[Structure], int], capacity: int, *, skipHeader: bool = False
) -> int:
    """
    Calculates the required size to store a queue of given size and item type

    Parameters
    ----------
    item: Structure | int
        Either a Structure describing a single item or the size of it
    capacity: int
        Maximum number of items the queue can hold
    skipHeader: bool, default=False
        True, if the queue should not contain a header
    """
    itemsize = item if isinstance(item, int) else sizeof(item)
    if skipHeader:
        return itemsize * capacity
    else:
        return sizeof(QueueView.Header) + itemsize * capacity


def as_queue(
    buffer: hp.Buffer,
    item: Type[Structure],
    *,
    offset: int = 0,
    size: Optional[int] = None,
    skipHeader: bool = False,
) -> QueueView:
    """
    Helper function returning a QueueView pointing at the given buffer.

    Parameters
    ----------
    buffer: hephaistos.Buffer
        Buffer containing the Queue
    item: Structure
        Structure describing a single item
    offset: int, default=0
        Offset in bytes into the buffer the view should start at
    size: int | None, default=None
        Size in bytes of the buffer the view should start.
        If None, the whole buffer minus offset is used.
    skipHeader: bool, default=False
        True, if data does not contain a queue header

    Returns
    -------
    view: QueueView
        The view describing the queue in the given buffer
    """
    if size is None:
        size = buffer.size_bytes - offset
    # calculate capacity
    if not skipHeader:
        size -= sizeof(QueueView.Header)
    item_size = sizeof(item)
    if size < 0 or size % item_size:
        raise ValueError("The size of the buffer does not match any queue size!")
    capacity = size // item_size
    # create view
    return QueueView(buffer.address, item, capacity, skipHeader=skipHeader)


class QueueBuffer(hp.RawBuffer):
    """
    Util class allocating enough memory to hold the given and gives a view to it

    Parameters
    ----------
    item: Structure
        Structure describing a single item
    capacity: int
        Maximum number of items the queue can hold
    skipHeader: bool, default=False
        True, if the queue should not contain a header
    """

    def __init__(
        self, item: Type[Structure], capacity: int, *, skipHeader: bool = False
    ) -> None:
        super().__init__(queueSize(item, capacity, skipHeader=skipHeader))
        self._view = QueueView(self.address, item, capacity, skipHeader=skipHeader)

    @property
    def count(self) -> int:
        """Number of items stored in the queue"""
        return self.view.count

    @property
    def view(self) -> QueueView:
        """View of the queue stored inside this buffer"""
        return self._view


class QueueTensor(hp.ByteTensor):
    """
    Util class allocating enough memory on the device to hold the given queue

    Parameters
    ----------
    item: Structure
        Structure describing a single item
    capacity: int
        Maximum number of items the queue can hold
    skipHeader: bool, default=False
        True, if the queue should not contain a header
    """

    def __init__(
        self, item: Type[Structure], capacity: int, *, skipHeader: bool = False
    ) -> None:
        super().__init__(queueSize(item, capacity, skipHeader=skipHeader))
        self._item = item
        self._capacity = capacity
        self._skipHeader = skipHeader

    @property
    def hasHeader(self) -> bool:
        """True if the queue has a header"""
        return not self._skipHeader

    @property
    def capacity(self) -> int:
        """Capacity of the queue"""
        return self._capacity

    @property
    def item(self) -> Type[Structure]:
        """Structure describing the items of the queue"""
        return self._item
