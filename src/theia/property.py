from __future__ import annotations

import hephaistos as hp

from ctypes import c_uint64
from dataclasses import dataclass
from itertools import product
from pathlib import Path, PurePath
from struct import pack, unpack
from zipfile import Path as ZipPath

from theia.lookup import Table

from collections.abc import Iterable, Iterator, Mapping
from importlib.resources.abc import Traversable
from types import MappingProxyType
from typing import ClassVar, Type
from typing_extensions import Self  # introduced in 3.11

__all__ = [
    "FloatProperty",
    "IntProperty",
    "TableProperty",
    "Property",
    "PropertyTable",
]


def __dir__():
    return __all__


@dataclass
class Property:
    """Contains the necessary information to populate a slot in a `MasterTable`"""

    data: int = 0
    """Data to be written in the master table. Must fit in 64 bits"""

    nbytes: int = 0
    """Amount of bytes to additionally allocate in the master table"""
    alignment: int = 4
    """Alignment of the additional allocation"""

    types: ClassVar[dict[str, Type[Property]]] = {}
    """Register mapping property types by their extension used for serialization"""
    extension: ClassVar[str] = ""
    """Extension used by this type of property"""

    def __init_subclass__(cls: Type[Property], *, ext: str) -> None:
        if not ext:
            raise RuntimeError("extension cannot be empty!")
        if ext in Property.types:
            raise RuntimeError(f'"{ext}" is already in use by another property type!')
        Property.types[ext] = cls
        cls.extension = ext

    def write(self, ptr: int, address: int) -> int:
        """
        Writes additional content at the given memory address at `ptr` and
        returns the actual written amount of bytes. `address` contains the
        corresponding memory address on GPU.

        Note
        ----
        No boundary checks are performed! Writing more than `nbytes` bytes
        results in undefined behavior!
        """
        return 0

    def save(self, file) -> None:
        """Writes property data into the given file"""
        file.write(pack("<Q", self.data))

    @classmethod
    def load(cls, file) -> Self:
        """Creates new property of this type using the data in the given file"""
        prop = cls()
        # TODO: read() does not guarantee to return exactly 8 bytes
        #       for now this should be fine though
        prop.data = unpack("<Q", file.read(8))[0]
        return prop

    @staticmethod
    def from_file(file: Traversable) -> tuple[str, Property]:
        """
        Loads the property from the given file. Uses the file extension to
        determine the correct property type. Returns tuple containing the name
        and property itself.
        """
        if not file.is_file():
            raise RuntimeError(f'Cannot load property from "{file.name}": Not a file!')
        # file is expected to have a name of: name.type, where type is the
        # extension specified by the subclasses of Property
        path = PurePath(file.name)  # Traversable has no stem and suffix property
        name, ext = path.stem, path.suffix
        if ext:
            # remove dot
            ext = ext[1:]
        if not ext:
            # if no extension present, default to base class
            cls = Property
        elif ext not in Property.types:
            raise RuntimeError(f'Unknown property type "{ext}"')
        else:
            cls = Property.types[ext]
        # delegate actual loading of property to the concrete class
        return (name, cls.load(file.open("rb")))


class IntProperty(Property, ext="int"):
    """Property containing two 32 bit integers"""

    def __init__(self, value: tuple[int, int] = (0, 0)) -> None:
        super().__init__()
        self.value = value

    @property
    def value(self) -> tuple[int, int]:
        """Values stored by this property"""
        return unpack("<ii", pack("<Q", self.data))

    @value.setter
    def value(self, value: tuple[int, int]) -> None:
        self.data = unpack("<Q", pack("<ii", *value))[0]


class FloatProperty(Property, ext="float"):
    """Property containing two 32 bit floats"""

    def __init__(self, value: float | tuple[float, float] = (0.0, 0.0)) -> None:
        super().__init__()
        if isinstance(value, float):
            value = (0.0, value)
        self.value = value

    @property
    def value(self) -> tuple[float, float]:
        """Values stored by this property"""
        return unpack("<ff", pack("<Q", self.data))

    @value.setter
    def value(self, value: tuple[float, float]) -> None:
        self.data = unpack("<Q", pack("<ff", *value))[0]


class TableProperty(Property, ext="table"):
    """
    Writes the given table into a property table and storing the address of
    it in the corresponding slot.

    Attributes
    ----------
    table: Table
        Table to be written in the properties table.
    """

    def __init__(self, table: Table | None = None) -> None:
        super().__init__(alignment=Table.ALIGNMENT)
        self.table = table

    @property
    def table(self) -> Table | None:
        return self._table

    @table.setter
    def table(self, value: Table | None) -> None:
        self._table = value
        if value is None or value.constant:
            self.nbytes = 0
        else:
            self.nbytes = value.nbytes

    def write(self, ptr: int, address: int) -> int:
        if self.table is None:
            self.data = 0
            return 0
        elif self.table.constant:
            # we can store the constant directly in the property table
            self.data = unpack("<Q", pack("<If", 0xF, self.table.samples.item()))[0]
            self.data = Table.createConstantValuePtr(self.table.samples.item())
            return 0
        else:
            self.data = address
            return self.table.copy(ptr)

    def save(self, file) -> None:
        if self.table is not None:
            self.table.save(file)

    @classmethod
    def load(cls, file) -> Self:
        return cls(Table.load(file))


PropertyTableEntry = Mapping[str, Property]
"""Single entry in a `PropertyTable`"""


def saveTableEntry(dir: Path | ZipPath, entry: PropertyTableEntry) -> None:
    """Saves the given property table entry in the specified directory"""
    if not dir.is_dir():
        raise RuntimeError(f"Expected a directory: {dir.name}")
    for name, prop in entry.items():
        filename = name + "." + prop.extension
        with dir.joinpath(filename).open("wb") as file:
            prop.save(file)


def saveTable(path: Path | ZipPath, table: Mapping[str, PropertyTableEntry]) -> None:
    """Saves the property table in the specified directory"""
    if not path.is_dir():
        raise RuntimeError(f'Expected a directory: "{path}"')
    for name, entry in table.items():
        entry_path = path.joinpath(f"{name}/")
        # we may need to create parent directories before creating files
        # e.g. in zip files this step is not necessary
        if hasattr(entry_path, "mkdir"):
            entry_path.mkdir(parents=True, exist_ok=True)

        saveTableEntry(entry_path, entry)


def loadTableEntry(dir: Traversable) -> PropertyTableEntry:
    """Loads the property table entry stored in the given directory"""
    if not dir.is_dir():
        raise RuntimeError(f'Expected a directory: "{dir.name}"')
    entry = {}
    # iterate over all files
    for file in dir.iterdir():
        name, prop = Property.from_file(file)
        entry[name] = prop
    return entry


def loadTable(dir: Traversable) -> Mapping[str, PropertyTableEntry]:
    """Loads all table entries stored in the given directory"""
    if not dir.is_dir():
        raise RuntimeError(f'Expected a directory: "{dir.name}"')
    return {d.name: loadTableEntry(d) for d in dir.iterdir()}


class PropertyTable:
    """
    Coordinates the storage of arbitrary data on the GPU organized in slots.
    One can imagine the master table as an ordinary table with the columns
    being the slots and the rows its entries. Each entry consists of a list of
    properties. The set of slots is implicitly defined by the union of the slots
    of all entries, which may differ from one another. Slots that are missing
    in some entries are automatically filled with zeros.

    Each slot has a fixed size of 64 bits, but a `Property` populating a slot
    can request additional memory in the master table and use the slot to store
    a pointer to it.

    Parameters
    ----------
    entries: dict[str, list[Property]]
        Dictionary mapping entry names to their list of properties. The names
        can later be used to retrieve the corresponding index or row number in
        the table.
    requiredSlots: Iterable[str], default=[]
        Set of required slots to be included in the table even if no entry
        contains it. The given order is preserved and will make up the first
        slots. For instance, the first entry in `requiredSlots` is guaranteed
        to be mapped to slot 0.
    """

    def __init__(
        self,
        entries: Mapping[str, PropertyTableEntry],
        *,
        requiredSlots: Iterable[str] | None = None,
    ) -> None:
        self._entries = dict(entries)
        self._entriesProxy = MappingProxyType(self._entries)
        self._indices = {name: i for i, name in enumerate(self._entries)}
        if requiredSlots is None:
            requiredSlots = []
        requiredSlots = list(requiredSlots)  # copy
        slots: frozenset[str] = frozenset(requiredSlots)
        for entry in self._entries.values():
            slots |= set(entry.keys())
        # create mapping slot -> slot index
        # ensure required slots come first
        # self._slots = { slot: i for i, slot in enumerate(slots) }
        self._slots = {slot: i for i, slot in enumerate(requiredSlots)}
        n = len(self._slots)
        missing = slots - set(requiredSlots)
        self._slots.update({slot: i for i, slot in enumerate(missing, n)})
        self._slotsProxy = MappingProxyType(self._slots)

        # calculate memory requirement
        # table starts with stride (one u64), followed by slots (one u64 each)
        tableSize = 1 + len(self._slots) * len(self._entries)
        nbytes = tableSize * 8
        align = lambda o, a: (a - (o % a)) % a  # (offset, alignment) -> padding
        for slot, entry in product(self._slots, self._entries.values()):
            if slot in entry:
                prop = entry[slot]
                nbytes += align(nbytes, prop.alignment)
                nbytes += prop.nbytes
        # allocate memory
        buffer = hp.Buffer(nbytes)
        self._tensor = hp.Tensor(nbytes)

        # skip table for now, we first want to write the additional data
        ptr = buffer.address
        adr = self._tensor.address
        offset = tableSize * 8
        for slot, entry in product(self._slots, self._entries.values()):
            if slot in entry:
                prop = entry[slot]
                offset += align(ptr + offset, prop.alignment)
                offset += prop.write(ptr + offset, adr + offset)
        assert offset == nbytes  # boundary check
        # write table itself at the beginning of buffer
        table = (c_uint64 * tableSize).from_address(buffer.address)
        table[0] = len(self._entries)  # stride between slots
        i = 1
        for slot, entry in product(self._slots, self._entries.values()):
            table[i] = entry[slot].data if slot in entry else 0
            i += 1
        assert i == tableSize  # boundary check
        # upload to GPU
        hp.execute(hp.updateTensor(buffer, self._tensor))

    @property
    def entries(self) -> MappingProxyType[str, PropertyTableEntry]:
        """Mapping of entry names to their respective index"""
        return self._entriesProxy

    @property
    def slots(self) -> MappingProxyType[str, int]:
        """Mapping of all slots present in the table to their corresponding index"""
        return self._slotsProxy

    @property
    def tensor(self) -> hp.Tensor:
        """Tensor containing the master table"""
        return self._tensor

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[str]:
        return iter(self._indices)

    def __contains__(self, item: str) -> bool:
        """Checks whether an entry of the given name exists in this table"""
        return item in self._indices

    def __getitem__(self, key: str) -> int:
        """
        Returns the index of the entry of the given name. Raises `KeyError`, if
        no such entry exists.
        """
        return self._indices[key]


def createSlotMacros(table: PropertyTable, prefix: str) -> dict[str, int]:
    """Creates macros for naming the slots of the given property table in shaders"""
    # we are rather naive here for now and assume all slot names behave nicely
    getMacroName = lambda slot: prefix + "_" + slot.replace(" ", "_").upper()
    return {getMacroName(slot): idx for slot, idx in table.slots.items()}
