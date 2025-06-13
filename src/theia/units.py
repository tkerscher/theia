from __future__ import annotations
from collections.abc import Iterable
from math import pi
from typing import Final, Literal, TypeVar


T = TypeVar("T")

# only export defined units
__all__ = [
    "convert",
    "km",
    "m",
    "cm",
    "mm",
    "inch",
    "ft",
    "yd",
    "mi",
    "per_km",
    "per_m",
    "per_cm",
    "per_mm",
    "c",
    "s",
    "ms",
    "us",
    "ns",
    "ps",
    "um",
    "nm",
    "rad",
    "deg",
    "eV",
    "keV",
    "GeV",
    "TeV",
    "PeV",
]


def __dir__():
    return sorted(
        __all__
        + [
            "setDimensionScales",
            "Dimension",
            "InverseUnit",
            "Unit",
        ],
        key=str.swapcase,
    )


Dimension = Literal["length", "time", "wavelength", "compound", "energy", "angle"]
"""
Fundamental dimensions the simulation knows about.
`compound` indicates a compound unit.
"""


class Unit:
    """
    Units allow to annotate numeric values with a notation of dimension and
    scaling while ensuring the scaling match between all values of the same
    dimension.

    Conceptually, `Unit` resembles the conversion factor from its designated
    unit to the internal one used by the library.

    Parameters
    ----------
    dimension: Dimension
        Dimension this unit applies to
    scale: float
        Scaling applied to get to unit dimension.

    Note
    ----
    Arithmetic with units are eagerly evaluated, thus changes to the global
    scaling via `setDimensionScales` wont apply to calculation made before.
    """

    # Thank you numpy for being weird
    # the following line forces (encourages) numpy to perform
    # multiplication / division not element-wise but pass the entire array to
    # __rmul__ and __rtruediv__
    # this also ensures to preserve the dtype of the array
    __array_priority__ = 9000

    def __init__(self, dimension: Dimension, scale: float = 1.0) -> None:
        self._dim = dimension
        self._scale = scale

    @property
    def dimension(self) -> Dimension:
        """Dimension this unit applies to"""
        return self._dim

    def __rmul__(self, value: T) -> T:
        # allow annotating tuples
        if type(value) is tuple:
            return tuple(i * self._scale for i in value)
        else:
            return value * self._scale

    def __rtruediv__(self, value: T) -> T:
        # allow annotating tuples
        if type(value) is tuple:
            return tuple(i / self._scale for i in value)
        else:
            return value / self._scale


class InverseUnit(Unit):
    """
    Util class for creating the inverse unit based on a regular unit, e.g. per
    meter. Equivalent to using the base unit while replacing multiplication
    with division and vice versa. Mainly serves as syntactic sugar.

    Parameters
    ----------
    unit: Unit
        Base unit to take the inverse from
    """

    def __init__(self, unit: Unit) -> None:
        super().__init__(unit.dimension)
        self._unit = unit

    def __rmul__(self, value: T) -> T:
        # this is the inverse, so we want to divide with the base unit instead
        return value / self._unit

    def __rtruediv__(self, value: T) -> T:
        # again, this is the inverse, so multiply with base unit instead
        return value * self._unit


class CompoundUnit(Unit):
    """
    Unit consisting of multiple base units, e.g. m/s.

    Parameters
    ----------
    const: float
        Constant factor of the unit
    nom: Iterable[Unit]
        Units in the nominator. Repeat the same unit to achieve powers
    denom: Iterable[Unit]
        Units in the denominator. Repeat the same unit to achieve powers
    """

    def __init__(
        self, const: float, nom: Iterable[Unit], denom: Iterable[Unit]
    ) -> None:
        super().__init__("compound")
        self._const = const
        self._nom = list(nom)
        self._denom = list(denom)

    @property
    def scale(self) -> float:
        """Value representing this unit"""
        scale = self._const
        for n in self._nom:
            scale *= n
        for d in self._denom:
            scale /= d
        return scale

    def __rmul__(self, value: T) -> T:
        # allow annotating tuples
        if type(value) is tuple:
            scale = self.scale  # cache scale
            return tuple(i * scale for i in value)
        else:
            return value * self.scale

    def __rtruediv__(self, value: T) -> T:
        # allow annotating tuples
        if type(value) is tuple:
            scale = self.scale  # cache scale
            return tuple(i / scale for i in value)
        else:
            return value / self.scale


# length
km: Final[Unit] = Unit("length", 1000.0)
m: Final[Unit] = Unit("length", 1.0)
cm: Final[Unit] = Unit("length", 0.01)
mm: Final[Unit] = Unit("length", 0.001)
# imperial length
inch: Final[Unit] = Unit("length", 0.0254)
ft: Final[Unit] = Unit("length", 0.3048)
yd: Final[Unit] = Unit("length", 0.9144)
mi: Final[Unit] = Unit("length", 1609.344)
# inverse length
per_km: Final[Unit] = InverseUnit(km)
per_m: Final[Unit] = InverseUnit(m)
per_cm: Final[Unit] = InverseUnit(cm)
per_mm: Final[Unit] = InverseUnit(mm)
# time
s: Final[Unit] = Unit("time", 1.0e9)
ms: Final[Unit] = Unit("time", 1.0e6)
us: Final[Unit] = Unit("time", 1.0e3)
ns: Final[Unit] = Unit("time", 1.0)
ps: Final[Unit] = Unit("time", 1.0e-3)
# speed
c: Final[Unit] = CompoundUnit(299792458.0, [m], [s])
# wavelength
um: Final[Unit] = Unit("wavelength", 1000.0)
nm: Final[Unit] = Unit("wavelength", 1.0)
# angle
rad: Final[Unit] = Unit("angle", 1.0)
deg: Final[Unit] = Unit("angle", pi / 180.0)
# energy
eV: Final[Unit] = Unit("energy", 1e-6)
keV: Final[Unit] = Unit("energy", 1e-3)
GeV: Final[Unit] = Unit("energy", 1.0)
TeV: Final[Unit] = Unit("energy", 1e3)
PeV: Final[Unit] = Unit("energy", 1e6)


def convert(data: T, unit: Unit) -> T:
    """
    Converts the given data to the specified unit.
    Syntactic sugar for `data / unit` making the intent more clear.
    """
    return data / unit
