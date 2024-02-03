import importlib.resources
import warnings

import hephaistos as hp
import numpy as np
from scipy.interpolate import CubicSpline

from ctypes import Structure, c_float, c_uint32, c_uint64, memmove, sizeof
from types import MappingProxyType

from theia.lookup import *

import numpy.typing as npt
from collections.abc import Iterable
from enum import IntFlag
from typing import Final, Union, Tuple


__all__ = [
    "parseMaterialFlags",
    "speed_of_light",
    "BK7Model",
    "FournierForandPhaseFunction",
    "HenyeyGreensteinPhaseFunction",
    "Material",
    "MaterialFlags",
    "MaterialStore",
    "Medium",
    "MediumModel",
    "SellmeierEquation",
    "serializeMedium",
    "WaterBaseModel",
]


def __dir__():
    return __all__


#################################### COMMON ####################################


speed_of_light: Final[float] = 0.299_792_458  # m / ns
"""speed of light in meter per nanosecond"""


class Medium:
    """
    Class to describe a named material by storing the physical properties as
    tables each containing values of its respective property sampled at
    equidistant positions on a specified range of wavelengths.

    Parameters
    ----------
    name: str
        Unique name of the material. Can later be used to fetch the address of
        the medium in gpu local memory.
    lambda_min: float
        Lower limit of the range of wavelengths the tables are defined
    lambda_max: float
        Upper limit of the range of wavelengths the tables are defined
    refractive_index: ArrayLike | None, default = None
        Table of refractive index as function of wavelength.
        None defaults to a constant value of 1.0.
    group_velocity: ArrayLike | None, default = None
        Table of group velocity in m/ns as function of wavelength.
        None defaults to a constant value of `speed_of_light`.
    absorption_coef: ArrayLike | None, default = None
        Table of absorption coefficient in units of 1/m as function of wavelength.
        None defaults to a constant value of 0.0.
    scattering_coef: ArrayLike | None, default = None
        Table of scattering coefficient in units of 1/m as function of wavelength.
        None defaults to a constant value of 0.0.
    log_phase_function: ArrayLike | None, default = None
        Table of logarithmic scattering phase function as a function of the
        cosine of the angle between incoming and outgoing ray in radians over
        the range [0,1].
        None defaults to a constant value of 1, i.e. uniform scattering.
    phase_sampling: ArrayLike | None, default = None
        Table containing values of the inverse cumulative density function of
        the phase function used for importance sampling.
        If None, sampling happens uniform random.
    """

    class GLSL(Structure):
        """The corresponding structure used in shaders"""

        _fields_ = [
            ("lambda_min", c_float),
            ("lambda_max", c_float),
            ("n", c_uint64),  # Table1D
            ("vg", c_uint64),  # Table1D
            ("mu_a", c_uint64),  # Table1D
            ("mu_s", c_uint64),  # Table1D
            ("log_phase", c_uint64),  # Table1D
            ("phase_sampling", c_uint64),  # Table1D
        ]

    ALIGNMENT: Final[int] = 8
    """Alignment of the GLSL structure in device memory"""

    def __init__(
        self,
        name: str,
        lambda_min: float,
        lambda_max: float,
        *,
        refractive_index: Union[npt.ArrayLike, None] = None,
        group_velocity: Union[npt.ArrayLike, None] = None,
        absorption_coef: Union[npt.ArrayLike, None] = None,
        scattering_coef: Union[npt.ArrayLike, None] = None,
        log_phase_function: Union[npt.ArrayLike, None] = None,
        phase_sampling: Union[npt.ArrayLike, None] = None,
    ) -> None:
        # store properties
        self.name = name
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.refractive_index = refractive_index
        self.group_velocity = group_velocity
        self.absorption_coef = absorption_coef
        self.scattering_coef = scattering_coef
        self.log_phase_function = log_phase_function
        self.phase_sampling = phase_sampling

    @property
    def name(self) -> str:
        """
        Name of this material. Can later be used to fetch the address of the
        medium in gpu local memory.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def lambda_min(self) -> float:
        """Lower end of the wavelength range on which the material properties are defined"""
        return self._lambda_min

    @lambda_min.setter
    def lambda_min(self, value: float) -> None:
        self._lambda_min = value

    @property
    def lambda_max(self) -> float:
        """Upper end of the wavelength range on which the material properties are defined"""
        return self._lambda_max

    @lambda_max.setter
    def lambda_max(self, value: float) -> None:
        self._lambda_max = value

    @property
    def refractive_index(self) -> Union[npt.ArrayLike, None]:
        """
        Table containing values of the refractive as a function of wavelength
        sampled at equidistant points on the range defined by lambda min/max.
        If None, a constant value of 1.0 is assumed.
        """
        return self._refractive_index

    @refractive_index.setter
    def refractive_index(self, value: Union[npt.ArrayLike, None]) -> None:
        self._refractive_index = value

    @refractive_index.deleter
    def refractive_index(self) -> None:
        self._refractive_index = None

    @property
    def group_velocity(self) -> Union[npt.ArrayLike, None]:
        """
        Table containing values of the group velocity in units of m/ns as a
        function of wavelength sampled at equidistant points on the range
        defined by lambda min/max.
        If None, a constant value of `speed_of_light` is assumed.
        """
        return self._group_velocity

    @group_velocity.setter
    def group_velocity(self, value: Union[npt.ArrayLike, None]) -> None:
        self._group_velocity = value

    @group_velocity.deleter
    def group_velocity(self) -> None:
        self._group_velocity = None

    @property
    def absorption_coef(self) -> Union[npt.ArrayLike, None]:
        """
        Table containing values of the absorption coefficient in units of 1/m
        as a function of wavelength sampled at equidistant points on the range
        defined by lambda min/max.
        If None, a constant value of 0.0 is assumed.
        """
        return self._absorption_coef

    @absorption_coef.setter
    def absorption_coef(self, value: Union[npt.ArrayLike, None]) -> None:
        self._absorption_coef = value

    @absorption_coef.deleter
    def absorption_coef(self) -> None:
        self._absorption_coef = None

    @property
    def scattering_coef(self) -> Union[npt.ArrayLike, None]:
        """
        Table containing values of the scattering coefficient in units of 1/m
        as a function of wavelength sampled at equidistant points on the range
        defined by lambda min/max.
        If None, a constant value of 0.0 is assumed.
        """
        return self._scattering_coef

    @scattering_coef.setter
    def scattering_coef(self, value: Union[npt.ArrayLike, None]) -> None:
        self._scattering_coef = value

    @scattering_coef.deleter
    def scattering_coef(self) -> None:
        self._scattering_coef = None

    @property
    def log_phase_function(self) -> Union[npt.ArrayLike, None]:
        """
        Table of logarithmic scattering phase function as a function of the
        cosine of the angle between incoming and outgoing ray in radians over
        the range [0,1].
        None defaults to a constant value of 1, i.e. uniform scattering.
        """
        return self._phase_function

    @log_phase_function.setter
    def log_phase_function(self, value: Union[npt.ArrayLike, None]) -> None:
        self._phase_function = value

    @log_phase_function.deleter
    def log_phase_function(self) -> None:
        self._phase_function = None

    @property
    def phase_sampling(self) -> Union[npt.ArrayLike, None]:
        """
        Table containing values of the inverse cumulative density function of
        the phase function used for importance sampling.
        If None, sampling happens uniform random.
        """
        return self._phase_sampling

    @phase_sampling.setter
    def phase_sampling(self, value: Union[npt.ArrayLike, None]) -> None:
        self._phase_sampling = value

    @phase_sampling.deleter
    def phase_sampling(self) -> None:
        self._phase_sampling = None


class MaterialFlags(IntFlag):
    """
    List of bit flags specifying the action a ray can take once it hit a
    material.
    """

    BLACK_BODY = 0x01
    """
    Material absorbs all rays completely and stops further tracing.
    Combined with `TARGET`, hits will not be attenuated by a transmission
    factor.
    """

    DETECTOR = 0x02
    """
    Marks the material as detector, i.e. tracers may only produce a response
    when hitting a material with this flag. Implies `NO_TRANSMIT`.

    Combine this flag with `BLACK_BODY` if reflectivity factors should not be
    applied.
    """

    NO_REFLECT = 0x08
    """
    Forbids tracer to reflect rays from the material, but still takes the
    reflectivity factor (i.e. Fresnel equation) into account.

    In combination with `NO_TRANSMIT` has the same effect as `BLACK_BODY`, but
    the latter should be preferred.
    """

    NO_TRANSMIT = 0x10
    """
    Forbids tracer to transmit rays through this material, but still takes the
    reflectivity factor (i.e. Fresnel equation) into account.

    In combination with `NO_REFLECT` has the same effect as `BLACK_BODY`, but
    the latter should be preferred.
    """

    VOLUME_BORDER = 0x20
    """
    Marks the boundary of a volume, where media changes but the rays neither
    reflect nor refract, i.e. keep straight. Can be used to model inhomogeneous
    media.
    """


_materialFlagsMap = {
    "B": MaterialFlags.BLACK_BODY,
    "D": MaterialFlags.DETECTOR,
    "R": MaterialFlags.NO_REFLECT,
    "T": MaterialFlags.NO_TRANSMIT,
    "V": MaterialFlags.VOLUME_BORDER,
}


def parseMaterialFlags(flags: str) -> MaterialFlags:
    """
    Parses the given string where each character represents a flag as listed
    below and returns the corresponding `MaterialFlags`. Capitalization does
    not matter. An empty string represents no flags.

    Note the flags `R`,`T` removes their corresponding flag, which are present
    at default.

    Flags:
     - `B` : `BLACK_BODY`
     - `D` : `DETECTOR`
     - `R` : removes `NO_REFLECT`
     - `T` : removes `NO_TRANSMIT`
     - `V` : `VOLUME_BORDER`
    """

    # edge case: empty input
    if len(flags) == 0:
        return MaterialFlags(0)

    # unify capitalization
    flags = flags.upper()

    # iterate characters
    result: MaterialFlags = MaterialFlags.NO_REFLECT | MaterialFlags.NO_TRANSMIT
    for flag in flags:
        if flag in _materialFlagsMap.keys():
            result ^= _materialFlagsMap[flag]
        else:
            raise ValueError(f"Unknown material flag '{flag}'")
    # done
    return result


class Material:
    """
    Class holding information about the material of a geometry. In general a
    geometry separates space into an "inside" and an "outside" defined by their
    normal vectors pointing outwards. Materials assign a Medium to each of them.

    Parameters
    ----------
    name: str
        Name of this Material. Can later be used to fetch the address of
        the material in gpu local memory.
    inside: Medium, str, None
        Medium in the inside of a geometry. Can also be specified by its name
        which will get resolved during baking/serialization.
        None defaults to vacuum.
    outside: Medium, str, None
        Medium in the outside of a geometry. Can also be specified by its name
        which will get resolved during baking/serialization.
        None defaults to vacuum.
    flags: (MaterialFlags|str,MaterialFlags|str)|MaterialFlags|str, default=0
        `MaterialFlags` applied to the material. In a tuple the first element
        applies to the inward direction, the second to the outward one.
        Otherwise the flags are applied to both directions. See
        `parseMaterialFlags` for a description on how to specify material flags
        with a string.
    """

    class GLSL(Structure):
        """The corresponding structure used in shaders"""

        _fields_ = [
            ("inside", c_uint64),  # buffer reference
            ("outside", c_uint64),  # buffer reference
            ("flagsInwards", c_uint32),
            ("flagsOutwards", c_uint32),
        ]

    ALIGNMENT: Final[int] = 8
    """Alignment of the GLSL structure in device memory"""

    def __init__(
        self,
        name: str,
        inside: Union[Medium, str, None],
        outside: Union[Medium, str, None],
        *,
        flags: (
            Tuple[MaterialFlags | str, MaterialFlags | str] | MaterialFlags | str
        ) = MaterialFlags(0),
    ) -> None:
        # store properties
        self.name = name
        self.inside = inside
        self.outside = outside
        if type(flags) == tuple:
            self.flagsInward = flags[0]
            self.flagsOutward = flags[1]
        else:
            self.flagsInward = flags
            self.flagsOutward = flags

    @property
    def name(self) -> str:
        """
        Name of this Material. Can later be used to fetch the address of
        the material in gpu local memory.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def inside(self) -> Union[Medium, str, None]:
        """
        Medium in the inside of a geometry. Can also be specified by its name
        which will get resolved during baking/serialization.
        None defaults to vacuum.
        """
        return self._inside

    @inside.setter
    def inside(self, value: Union[Medium, str, None]) -> None:
        self._inside = value

    @property
    def outside(self) -> Union[Medium, str, None]:
        """
        Medium in the outside of a geometry. Can also be specified by its name
        which will get resolved during baking/serialization.
        None defaults to vacuum.
        """
        return self._outside

    @outside.setter
    def outside(self, value: Union[Medium, str, None]) -> None:
        self._outside = value

    @property
    def flagsOutward(self) -> MaterialFlags:
        """
        Material flags specifying the behavior of rays hitting the material from
        the `inside` medium
        """
        return self._flagsOutward

    @flagsOutward.setter
    def flagsOutward(self, value: MaterialFlags | str) -> None:
        if type(value) != MaterialFlags:
            value = parseMaterialFlags(value)
        self._flagsOutward = value

    @flagsOutward.deleter
    def flagsOutward(self) -> None:
        self._flagsOutward = MaterialFlags(0)

    @property
    def flagsInward(self) -> MaterialFlags:
        """
        Material flags specifying the behavior of rays hitting the material from
        the `inside` medium
        """
        return self._flagsInward

    @flagsInward.setter
    def flagsInward(self, value: MaterialFlags | str) -> None:
        if type(value) != MaterialFlags:
            value = parseMaterialFlags(value)
        self._flagsInward = value

    @flagsInward.deleter
    def flagsInward(self) -> None:
        self._flagsInward = MaterialFlags(0)


class MaterialStore:
    """
    Manages the upload and lifetime of media and material on the device, while
    providing an easy interface for querying their device addresses.
    Optionally allows to update existing material and media as long as no
    entirely new data is added, e.g. populating a previously empty property or
    increasing the sample count of an existing table.
    """

    def __init__(
        self,
        material: Iterable[Material],
        *,
        media: Iterable[Medium] = [],
        freeze: bool = True,
    ) -> None:
        # during initial construction, it's always unfrozen
        self._frozen = False

        # start by virtually allocating everything (i.e. store offsets)
        size = 0

        def alloc(n: int, alignment: int) -> int:
            nonlocal size
            if size % alignment != 0:
                size += alignment - (size % alignment)
            ptr = size
            size += n
            return ptr

        # alloc methods
        self._table_ptr: dict[str, Tuple[int, int]] = {}  # (ptr, size)
        self._media_ptr: dict[str, int] = {}
        self._mat_ptr: dict[str, int] = {}

        def allocTable(name: str, data):
            # assume we only add new tables
            if data is not None:
                size = getTableSize(data)
                self._table_ptr[name] = (alloc(size, TABLE_ALIGNMENT), size)

        def allocMedium(medium: Medium | str | None):
            if medium is None:
                return
            name = medium.name if type(medium) == Medium else str(medium)
            if name in self._media_ptr:
                return
            self._media_ptr[name] = alloc(sizeof(Medium.GLSL), Medium.ALIGNMENT)
            allocTable(f"{name}_n", medium.refractive_index)
            allocTable(f"{name}_vg", medium.group_velocity)
            allocTable(f"{name}_mua", medium.absorption_coef)
            allocTable(f"{name}_mus", medium.scattering_coef)
            allocTable(f"{name}_lpf", medium.log_phase_function)
            allocTable(f"{name}_ps", medium.phase_sampling)

        def allocMaterial(mat: Material):
            if mat.name in self._mat_ptr:
                return
            self._mat_ptr[mat.name] = alloc(sizeof(Material.GLSL), Material.ALIGNMENT)
            allocMedium(mat.inside)
            allocMedium(mat.outside)

        # virtually allocate everything
        for medium in media:
            allocMedium(medium)
        for mat in material:
            allocMaterial(mat)

        # allocate actual memory
        self._tensor = hp.ByteTensor(size, mapped=(not freeze))
        self._buffer = hp.RawBuffer(size)
        # calculate device addresses
        adr = self._tensor.address
        self._table_adr = {t: adr + d for t, (d, _) in self._table_ptr.items()}
        self._media_adr = {m: adr + offset for m, offset in self._media_ptr.items()}
        self._mat_adr = {mat: adr + offset for mat, offset in self._mat_ptr.items()}
        # promote offsets to pointers into staging buffer
        ptr = self._buffer.address
        self._table_ptr = {t: (ptr + d, s) for t, (d, s) in self._table_ptr.items()}
        self._media_ptr = {m: ptr + offset for m, offset in self._media_ptr.items()}
        self._mat_ptr = {mat: ptr + offset for mat, offset in self._mat_ptr.items()}

        # create read only proxy on device addresses
        self._media = MappingProxyType(self._media_adr)
        self._mat = MappingProxyType(self._mat_adr)

        # finally, write some actual data
        processed_media: set[str] = set()

        def procMedium(medium: Medium | None):
            if medium is not None and medium.name not in processed_media:
                self.updateMedium(medium)
                processed_media.add(medium.name)

        for medium in media:
            procMedium(medium)
        for mat in material:
            self.updateMaterial(mat, updateMedia=False)
            procMedium(mat.inside)
            procMedium(mat.outside)

        # upload data to tensor
        if freeze:
            # tensor is not mapped if we're going to freeze it
            # -> let Vulkan copy it
            hp.execute(hp.updateTensor(self._buffer, self._tensor, unsafe=True))
        else:
            self.flush()
        # freeze if necessary
        self._frozen = freeze

    @property
    def frozen(self) -> bool:
        """Wether this MaterialStore is frozen, i.e. does not support updates"""
        return self._frozen

    @property
    def material(self) -> MappingProxyType[str, int]:
        """
        Read only map of all registered material returning their device address
        as used by programs by their name.
        """
        return self._mat

    @property
    def media(self) -> MappingProxyType[str, int]:
        """
        Read only map of all registered media returning their device address as
        used by programs by their name.
        """
        return self._media

    def flush(self) -> None:
        """
        Flushes local updates to the device.
        Does nothing if the store is frozen.
        """
        if not self.frozen:
            self._tensor.update(self._buffer.address, self._buffer.size_bytes)

    def _updateTable(self, name: str, data) -> int:
        """Internal fn for updating tables"""
        if data is None:
            return 0
        if name not in self._table_ptr:
            raise ValueError(f"Table {name} has not been previously allocated")
        ptr, size = self._table_ptr[name]
        data_size = getTableSize(data)
        if data_size > size:
            raise ValueError(f"Table {name} does not fit in previous allocation")
        table = createTable(data)
        memmove(ptr, table.ctypes.data, data_size)
        return self._table_adr[name]

    def updateMedium(self, medium: Medium) -> None:
        """
        Updates the internal representation of the given medium with new data.
        Call flush() to upload changes to the device. Fails if the store is
        frozen.
        """
        if self.frozen:
            raise RuntimeError("Cannot update frozen MaterialStore")
        if medium.name not in self._media_ptr:
            raise ValueError(f"Medium {medium.name} has not been previously allocated")
        # save header
        glsl = Medium.GLSL.from_address(self._media_ptr[medium.name])
        glsl.lambda_min = medium.lambda_min
        glsl.lambda_max = medium.lambda_max
        # save tables
        glsl.n = self._updateTable(f"{medium.name}_n", medium.refractive_index)
        glsl.vg = self._updateTable(f"{medium.name}_vg", medium.group_velocity)
        glsl.mu_a = self._updateTable(f"{medium.name}_mua", medium.absorption_coef)
        glsl.mu_s = self._updateTable(f"{medium.name}_mus", medium.scattering_coef)
        glsl.log_phase = self._updateTable(
            f"{medium.name}_lpf", medium.log_phase_function
        )
        glsl.phase_sampling = self._updateTable(
            f"{medium.name}_ps", medium.phase_sampling
        )

    def updateMaterial(self, material: Material, *, updateMedia: bool = False) -> None:
        """
        Updates the internal representation of the given material with new data.
        Call flush() to upload changes to the device. Fails if the store is
        frozen.

        Parameters
        ----------
        material: Material
            Material to be updated containing new data
        updateMedia: bool, default=False
            Wether to also update referenced media.
        """
        if self.frozen:
            raise RuntimeError("Cannot update frozen MaterialStore")
        if material.name not in self._mat_ptr:
            raise ValueError(
                f"Material {material.name} has not been previously allocated"
            )
        inside, outside = material.inside, material.outside
        if inside is not None and inside.name not in self.media:
            raise ValueError(
                f"Material {material.name} references unknown material {inside.name}"
            )
        if outside is not None and outside.name not in self.media:
            raise ValueError(
                f"Material {material.name} references unknown material {outside.name}"
            )
        # fetch header
        glsl = Material.GLSL.from_address(self._mat_ptr[material.name])
        glsl.inside = 0 if inside is None else self.media[inside.name]
        glsl.outside = 0 if outside is None else self.media[outside.name]
        glsl.flagsInwards = material.flagsInward
        glsl.flagsOutwards = material.flagsOutward


#################################### MODELS ####################################


class MediumModel:
    """
    Base class for models of media. Implements a function for creating a medium
    by sampling functions provided by base classes.
    """

    # To be overwritten in base classes
    ModelName = "noname"

    def refractive_index(self, wavelength: npt.ArrayLike) -> Union[npt.NDArray, None]:
        """
        Calculates the refractive index for the given wavelengths in nm.
        Returns None if not defined.
        """
        return None

    def group_velocity(self, wavelength: npt.ArrayLike) -> Union[npt.NDArray, None]:
        """
        Calculates the group velocity in m/ns for the given wavelengths in nm.
        Returns None if not defined.
        """
        return None

    def absorption_coef(self, wavelength: npt.ArrayLike) -> Union[npt.NDArray, None]:
        """
        Returns the absorption coefficient in units 1/m for the given
        wavelengths in nm.
        Returns None if not defined.
        """
        return None

    def scattering_coef(self, wavelength: npt.ArrayLike) -> Union[npt.NDArray, None]:
        """
        Returns the scattering coefficient in units 1/m for the given
        wavelengths in nm.
        Returns None if not defined.
        """
        return None

    def log_phase_function(self, cos_theta: npt.ArrayLike) -> Union[npt.NDArray, None]:
        """
        Evaluates the log phase functions for the given values of cos theta
        with theta being the angle between incoming and outgoing ray after
        scattering.
        Returns None if not defined.
        """
        return None

    def phase_sampling(self, eta: npt.ArrayLike) -> Union[npt.NDArray, None]:
        """
        Returns samples of cos(theta) using the uniform unit random numbers eta
        which follows the phase function as underlying distribution.
        Returns None if not defined.
        """
        return None

    def createMedium(
        self,
        lambda_min=200.0,
        lambda_max=800.0,
        num_lambda=1024,
        num_theta=1024,
        *,
        name: Union[str, None] = None,
    ) -> Medium:
        """
        Creates a medium from this model over the given range of wavelengths
        in nm.

        Parameters
        ----------
        lambda_min: float, default=200.0 [nm]
            Lower limit of the range of wavelengths to be sampled
        lambda_max: float, default=800.0 [nm]
            Upper limit of the range of wavelengths to be sampled
        num_lambda: int, default=256
            Number of samples to take from the wavelength range
        num_theta: int, default=256
            Number of samples to take from the phase and its sample function
        name: str | None, default is model name
            Explicit name for the new created model
        """
        l = np.linspace(lambda_min, lambda_max, num_lambda)
        t = np.linspace(-1.0, 1.0, num_theta)  # cos(theta)
        e = np.linspace(0.0, 1.0, num_theta)
        return Medium(
            name if name is not None else self.ModelName,
            lambda_min,
            lambda_max,
            refractive_index=self.refractive_index(l),
            group_velocity=self.group_velocity(l),
            absorption_coef=self.absorption_coef(l),
            scattering_coef=self.scattering_coef(l),
            log_phase_function=self.log_phase_function(t),
            phase_sampling=self.phase_sampling(e),
        )


class SellmeierEquation:
    """
    The Sellmeier equation is a empirical model of the refractive index as a
    function of the wavelength parameterized by a set of 6 constants for some
    transparent media.

    .. math::
        n^2(\\lambda) = 1 + \\sum_i\\frac{B_i\\lambda^2}{\\lambda^2-C_i}

    See https://en.wikipedia.org/wiki/Sellmeier_equation
    """

    def __init__(
        self, B1: float, B2: float, B3: float, C1: float, C2: float, C3: float
    ) -> None:
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3

    def refractive_index(self, wavelength: npt.ArrayLike) -> npt.NDArray:
        """Calculates the refractive index for the given wavelengths in nm"""
        L2 = np.square(wavelength)
        S1 = self.B1 * L2 / (L2 - self.C1)
        S2 = self.B2 * L2 / (L2 - self.C2)
        S3 = self.B3 * L2 / (L2 - self.C3)
        return np.sqrt(1.0 + S1 + S2 + S3)

    def group_velocity(self, wavelength: npt.ArrayLike) -> npt.NDArray:
        """
        Calculates the group velocity in m/ns for the given wavelengths in nm
        """
        n = self.refractive_index(wavelength)
        L = wavelength
        L2 = np.square(wavelength)
        S1 = self.B1 * self.C1 * L / np.square(L2 - self.C1)
        S2 = self.B2 * self.C2 * L / np.square(L2 - self.C2)
        S3 = self.B3 * self.C3 * L / np.square(L2 - self.C3)
        grad = -1.0 * (S1 + S2 + S3) / n
        return speed_of_light / (n - wavelength * grad)


class BK7Model(SellmeierEquation, MediumModel):
    """
    Model for BK7 glass based on the Sellmeier equation for refractive index
    and measurements for the absorption coefficient.

    See https://www.schott.com/shop/advanced-optics/en/Optical-Glass/SCHOTT-N-BK7/c/glass-SCHOTT%20N-BK7%C2%AE
    """

    ModelName = "bk7"

    # data table will loaded on first model init
    TransmissionTable = None

    def __init__(self) -> None:
        super().__init__(
            1.03961212,  # B1
            0.231792344,  # B2
            1.010469450,  # B3
            0.00600069867e6,  # C1 [nm^2]
            0.0200179144e6,  # C2 [nm^2]
            103.5606530e6,  # C3 [nm^2]
        )
        # lazily load data
        if BK7Model.TransmissionTable is None:
            BK7Model.TransmissionTable = np.loadtxt(
                importlib.resources.files("theia").joinpath(
                    "data/bk7_transmission.csv"
                ),
                delimiter=",",
                skiprows=2,
            )

    def absorption_coef(self, wavelength: npt.ArrayLike) -> npt.NDArray:
        """
        Returns the absorption coefficient in units 1/m for the given
        wavelengths in nm
        """
        # we can transform the transmission measurements to absorption
        # coefficients via the Beer-Lambert law. Unfortunately, they two
        # measurements disagree a lot. So we take the average weighted by
        # the probe thickness, as thicker ones should give a better result
        # To avoid taking the average with inf, we actually take the average
        # of the absorption lengths, i.e. inf -> 0

        mu = None
        # disable error, since we'll take the log of zero
        with np.errstate(divide="ignore"):
            tau_10mm = -0.010 / np.log(BK7Model.TransmissionTable[:, 1])
            tau_25mm = -0.025 / np.log(BK7Model.TransmissionTable[:, 2])
            tau_avg = (10.0 * tau_10mm + 25.0 * tau_25mm) / 35.0
            # interpolate without inf
            tau = np.interp(wavelength, BK7Model.TransmissionTable[:, 0], tau_avg)
            # convert back to coefficients
            return np.reciprocal(tau)


class HenyeyGreensteinPhaseFunction:
    """
    Models the phase function proposed by Henyey and Greenstein [HG41].

    Parameters
    ----------
    g: float
        Asymmetry parameter controlling the distribution

    See
    ---
    [HG41] Henyey, L. G., and J. L. Greenstein. 1941. Diffuse radiation in the galaxy. Astrophysical Journal 93, 70-83
    """

    def __init__(self, g: float = 0.0) -> None:
        self.g = g

    @property
    def g(self) -> float:
        """Asymmetry parameter controlling the distribution"""
        return self._g

    @g.setter
    def g(self, value: float) -> None:
        self._g = value
        if not -1.0 < value < 1.0:
            warnings.warn(
                "Asymmetry parameter outside the valid range (-1,1)!", RuntimeWarning
            )

    def log_phase_function(self, cos_theta: npt.ArrayLike) -> npt.NDArray:
        """
        Evaluates the log phase function for the given angles as cos(theta).
        Normalized with respect to unit sphere.
        """
        return np.log(
            (1.0 - self.g**2)
            / np.power(1.0 + self.g**2 - 2 * self.g * cos_theta, 1.5)
            / (4.0 * np.pi)
        )

    def phase_sampling(self, eta: npt.ArrayLike) -> npt.ArrayLike:
        """
        Samples the phase function using provided unit random numbers eta.
        Returns the cosine of the sampled angle.

        See Zhang, J.: On Sampling of Scattering Phase Functions, 2019
        """
        if abs(self.g) < 1e-7:
            # prevent division by zero: g=0 -> uniform
            return 1.0 - 2.0 * eta
        else:
            return (
                1.0
                + self.g**2
                - ((1.0 - self.g**2) / (1 + self.g - 2.0 * self.g * eta)) ** 2
            ) / (2.0 * self.g)


class FournierForandPhaseFunction:
    """
    Approximation of scattering in an ensemble of spherical particles with an
    refractive index n following a hyperbolic (Junge) size distribution
    N(r) ~ r^(-mu) where r is the radius and mu the slope of log N(r).

    See Fournier, G. and M. Jonasz, 1999. Computer based underwater imaging
        analysis. In Airborne and In-water Underwater Imaging,
        SPIE Vol. 3761, G. Gilbert [ed], 62-77 (with corrections)
    """

    def __init__(self, n: float, mu: float) -> None:
        self.n = n
        self.mu = mu

        # Unfortunately this phase function is rather complex
        # While there exist a analytic integral, it's hard to invert so we'll
        # just evaluate the integral and interpolate the inverse

        # sample phase function integral
        # note that the phase function diverges for cos(theta) = 1
        # that's not that big of a problem, since rng never produces exactly 1
        cos_theta = np.linspace(1.0 - 1e-7, -1.0, 2048)  # TODO: tune number of eval
        # some constants
        nu = 0.5 * (3.0 - self.mu)
        d = 2.0 * (1.0 - cos_theta) / (3.0 * (self.n - 1.0) ** 2)
        d_nu = np.float_power(d, nu)
        d_180 = 4.0 / (3.0 * (self.n - 1.0) ** 2)
        d_180_nu = np.float_power(d_180, nu)
        # split up formula to be more readable
        A = ((1 - d_nu * d) - 0.5 * (1 - d_nu) * (1 - cos_theta)) / ((1 - d) * d_nu)
        B = ((1 - d_180_nu) * (1 - cos_theta) * cos_theta) / (
            16 * (d_180 - 1) * d_180_nu
        )
        cdf = A + B
        # fill interpolator
        self._sample_spline = CubicSpline(cdf, cos_theta)

    @property
    def n(self) -> float:
        """The refractive index of the particles"""
        return self._n

    @n.setter
    def n(self, value: float) -> None:
        self._n = value

    @property
    def mu(self) -> float:
        """The slope of the particle size distribution log N(r) ~ -mu*log(r)"""
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        self._mu = value

    def log_phase_function(self, cos_theta: npt.ArrayLike) -> npt.NDArray:
        """Evaluates the log phase function for the given angles mu = cos(theta)"""
        # constants
        nu = 0.5 * (3.0 - self.mu)
        d = 2.0 * (1.0 - cos_theta) / (3.0 * (self.n - 1.0) ** 2)
        d_nu = np.float_power(d, nu)
        d_180 = 4.0 / (3.0 * (self.n - 1.0) ** 2)
        d_180_nu = np.float_power(d_180, nu)
        # formula split to be more readable
        x = cos_theta
        A = nu * (1 - d) - (1 - d_nu) + 2 * (d * (1 - d_nu) - nu * (1 - d)) / (1 - x)
        B = 4 * np.pi * (1 - d) ** 2 * d_nu
        C = (1 - d_180_nu) * (3 * x**2 - 1)
        D = 16 * np.pi * (d_180 - 1) * d_180_nu
        return np.log(A / B + C / D)

    def phase_sampling(self, eta: npt.ArrayLike) -> npt.NDArray:
        """
        Samples the phase function using provided unit random numbers eta.
        Returns the cosine of the sampled angle.
        """
        # we already did the heavy lifting -> just evaluate
        # note that since the spline extrapolates, it can produce values
        # outside [-1,1]. We expect the sampler to clamp so this actually helps
        # to be more accurate during linear interpolating the sampling
        # Also note that thanks to extrapolation, the divergence at
        # cos(theta)=1.0 is no longer a problem
        return self._sample_spline(eta)


class WaterBaseModel:
    """
    Base model for calculating the optical properties of (sea) water laking the
    phase function. Calculations of the refractive index follows [MS90], which
    proposed a fit on empiric data. The absorption and scattering model is
    based on empiric data from [Smith81].

    Parameters
    ----------
    temperature: float, [°C]
        water temperature
    pressure: float, [dbar]
        water pressure in dbar
    salinity: float, [PSU]
        water salinity in practical salinity units (PSU)
    """

    # lazily loaded data table
    DataTable = None

    # constants
    A0 = 1.3280657
    L2 = -0.0045536802
    LM2 = 0.0025471707
    LM4 = 0.000007501966
    LM6 = 0.000002802632
    T1 = -0.0000052883907
    T2 = -0.0000030738272
    T3 = 0.000000030124687
    T4 = -2.0863178e-10
    TL = 0.000010508621
    T2L = 0.00000021282248
    T3L = -0.000000001705881
    S0 = 0.00019029121
    S1LM2 = 0.0000024239607
    S1T = -0.00000073960297
    S1T2 = 0.0000000089818478
    S1T3 = 1.2078804e-10
    STL = -0.0000003589495
    P1 = 0.0000015868363
    P2 = -1.574074e-11
    PLM2 = 0.000000010712063
    PT = -0.0000000094634486
    PT2 = 1.0100326e-10
    P2T2 = 5.8085198e-15
    P1S = -0.0000000011177517
    PTS = 5.7311268e-11
    PT2S = -1.5460458e-12

    def __init__(self, temperature: float, pressure: float, salinity: float) -> None:
        self.temperature = temperature
        self.pressure = pressure
        self.salinity = salinity
        # check if we need to load the data
        if WaterBaseModel.DataTable is None:
            WaterBaseModel.DataTable = np.loadtxt(
                importlib.resources.files("theia").joinpath("data/water_smith81.csv"),
                delimiter=",",
                skiprows=1,
            )

    @property
    def temperature(self) -> float:
        """Water temperature in °C"""
        return self._T

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._T = value
        if not 0.0 <= value <= 30.0:
            warnings.warn(
                "Temperature is outside the models valid range of 0°-30°C",
                RuntimeWarning,
            )

    @property
    def pressure(self) -> float:
        """Water pressure in dbar"""
        return self._p

    @pressure.setter
    def pressure(self, value: float) -> None:
        self._p = value
        if not 0.0 <= value <= 11_000:
            warnings.warn(
                "Pressure is outside the models valid range of 0-11.000 dbar",
                RuntimeWarning,
            )

    @property
    def salinity(self) -> float:
        """Salinity measured in practical salinity units (PSU)"""
        return self._S

    @salinity.setter
    def salinity(self, value: float) -> None:
        self._S = value
        if not 0.0 <= value <= 40.0:
            warnings.warn(
                "Salinity is outside the models valid range of 0-40 psu", RuntimeWarning
            )

    def refractive_index(self, wavelength: npt.ArrayLike) -> npt.NDArray:
        """Calculates the refractive index for the given wavelengths"""
        # formula expects wavelengths in micrometers -> convert
        L = wavelength / 1000.0
        T = self.temperature
        p = self.pressure
        S = self.salinity
        # following naming convention from paper
        N1 = (
            self.A0
            + self.L2 * (L**2)
            + self.LM2 / (L**2)
            + self.LM4 / (L**4)
            + self.LM6 / (L**6)
            + self.T1 * T
            + self.T2 * (T**2)
            + self.T3 * (T**3)
            + self.T4 * (T**4)
            + self.TL * T * L
            + self.T2L * (T**2) * L
            + self.T3L * (T**3) * L
        )
        N2 = (
            self.S0 * S
            + self.S1LM2 * S / (L**2)
            + self.S1T * S * T
            + self.S1T2 * S * (T**2)
            + self.S1T3 * S * (T**3)
            + self.STL * S * T * L
        )
        N3 = (
            self.P1 * p
            + self.P2 * (p**2)
            + self.PLM2 * p / (L**2)
            + self.PT * p * T
            + self.PT2 * p * (T**2)
            + self.P2T2 * (p**2) * (T**2)
        )
        N4 = self.P1S * p * S + self.PTS * p * T * S + self.PT2S * p * (T**2) * S
        return N1 + N2 + N3 + N4

    def group_velocity(self, wavelength: npt.ArrayLike) -> npt.NDArray:
        """
        Calculates the group velocity in m/ns for the given wavelengths in nm
        """
        # formula expects wavelengths in micrometers -> convert
        L = wavelength / 1000.0
        T = self.temperature
        p = self.pressure
        S = self.salinity
        # G_i = dN_i/dL
        G1 = (
            2.0 * self.L2 * L
            - 2.0 * self.LM2 / (L**3)
            - 4.0 * self.LM4 / (L**5)
            - 6.0 * self.LM6 / (L**7)
            + self.TL * T
            + self.T2L * (T**2)
            + self.T3L * (T**3)
        )
        G2 = -2.0 * self.S1LM2 * S / (L**3) + self.STL * S * T
        G3 = -2.0 * self.PLM2 * p / (L**3)
        G4 = 0.0
        G = G1 + G2 + G3 + G4
        # vg = c / (n - L*dn/dL)
        n = self.refractive_index(wavelength)
        return speed_of_light / (n - L * G)

    def absorption_coef(self, wavelength: npt.ArrayLike) -> npt.NDArray:
        """
        Returns the absorption coefficient in units 1/m for the given
        wavelengths in nm
        """
        return np.interp(
            wavelength, WaterBaseModel.DataTable[:, 0], WaterBaseModel.DataTable[:, 1]
        )

    def scattering_coef(self, wavelength: npt.ArrayLike) -> npt.NDArray:
        """
        Returns the scattering coefficient in units 1/m for the given
        wavelengths in nm
        """
        return np.interp(
            wavelength, WaterBaseModel.DataTable[:, 0], WaterBaseModel.DataTable[:, 2]
        )
