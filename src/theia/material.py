from __future__ import annotations

import re
import hephaistos as hp
import numpy as np
import scipy.constants
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.stats.sampling import NumericalInversePolynomial
import warnings

from dataclasses import dataclass, field
from enum import Enum, IntFlag
from itertools import chain
from pathlib import Path
from struct import pack, unpack
from types import SimpleNamespace
from zipfile import ZipFile, Path as ZipPath

from theia.lookup import createTableFromFunction, Interpolation
from theia.property import (
    FloatProperty,
    Property,
    PropertyTable,
    PropertyTableEntry,
    TableProperty,
    createSlotMacros,
    loadTable,
    loadTableEntry,
    saveTable,
    saveTableEntry,
)
from theia.util import classproperty, createPreamble, intersectRange, loadCSV
import theia.units as u

from collections.abc import Callable, Iterable, Iterator, Mapping
from numpy.typing import ArrayLike, NDArray
from typing import Any, Final, Generic, TypeVar, overload
from typing_extensions import Self


__all__ = [
    "getPropertySamples",
    "loadMaterials",
    "medium_property",
    "saveMaterials",
    "speed_of_light",
    "BK7Model",
    "DefaultMaterialSlots",
    "DefaultMediaSlots",
    "DispersionFreeMedium",
    "FournierForandPhaseFunction",
    "HenyeyGreensteinPhaseFunction",
    "KokhanovskyOceanWaterPhaseMatrix",
    "Material",
    "MaterialFlags",
    "MaterialStore",
    "Medium",
    "MediumModel",
    "NumericalPhaseSamplingMixin",
    "OpticalProperties",
    "parseMaterialFlags",
    "PureWaterModel",
    "RayleighScatteringModel",
    "SellmeierEquation",
    "SmithNaturalWaterMeasurements",
    "VACUUM_IDX",
    "WaterBaseModel",
]


def __dir__():
    return __all__


#################################### COMMON ####################################


speed_of_light: Final[float] = 1.0 * u.c
"""speed of light in internal units"""


@dataclass
class Medium:
    """
    Describes a named medium by its physical properties stored as `Property`.
    This allows for an arbitrary set of physical properties consisting of
    various data types such as constants or look up tables.
    """

    name: str
    """Name of the medium"""
    wavelengthRange: tuple[float, float]
    """Wavelength range for which the optical properties are defined"""
    properties: dict[str, Property] = field(default_factory=dict)
    """Optical properties describing the medium"""


def _createTableEntryFromMedium(medium: Medium) -> PropertyTableEntry:
    """util function converting Medium to a property table entry"""
    props: dict[str, Property]
    props = {"wavelength_range": FloatProperty(medium.wavelengthRange)}
    props.update(medium.properties)
    return props


class OpticalProperties(str, Enum):
    """Enumeration of known optical properties"""

    ABSORPTION_COEF = "absorption_coef"
    GROUP_VELOCITY = "group_velocity"
    LOG_PHASE_FUNC = "log_phase_function"
    PHASE_SAMPLING = "phase_sampling"
    REFRACTIVE_INDEX = "refractive_index"
    SCATTERING_COEF = "scattering_coef"
    # polarization
    PHASE_M12 = "phase_m12"
    PHASE_M22 = "phase_m22"
    PHASE_M33 = "phase_m33"
    PHASE_M34 = "phase_m34"


def checkMedium(
    medium: Medium, *, raises: bool = True, rtol: float = 5e-3, atol: float = 1e-5
) -> bool:
    """
    Runs a series of basic tests on the given medium for physical plausibility
    and returns `True` if it is successfull. On failure, if `raises` is `True`
    raises an `AssertionError` or returns `False` otherwise. Note, that only
    known properties will be checked, but `medium` may contain additional ones.
    See `OpticalProperties` for an enumeration of known optical properties.

    Parameters
    ----------
    medium: Medium
        Medium to check
    raises: bool, default=True
        Whether to raise an error on failure
    rtol: float, default=5e-3
        Relative tolerance when comparing numerical estimated values to medium
    atol: float, default=1e-5
        Absolute tolerance when comparing numerical estimated values to medium

    Note
    ----
    If the medium is only sparsely sampled you might need to increase the
    tolerances. This does not necessarily mean that the model is wrong, but it
    might be worth to double check.
    """

    # common logic checking for table properties and fetching samples
    def getTableData(name: str) -> NDArray | None:
        if name not in medium.properties:
            return None
        prop = medium.properties[name]
        assert isinstance(
            prop, TableProperty
        ), f'Expected property "{name}" to be a table!'
        if prop.table is None:
            return None
        data = prop.table.samples
        assert data.ndim == 1, f'Expected data of property "{name}" to be 1D!'
        return data

    # bit ugly, but nicer than duplicating the same if/else over and over
    try:
        # fmt: off
        lamMin, lamMax = medium.wavelengthRange
        assert lamMax >= lamMax
        if (n := getTableData("refractive_index")) is not None:
            assert "group_velocity" in medium.properties, "Refractive index was specified, but no group velocity!"
            assert np.min(n) >= 1.0, "Refractive index must be at least 1.0!"
        if (vg := getTableData("group_velocity")) is not None:
            assert "refractive_index" in medium.properties, "Group velocity was specified, but no refractive index!"
            n = getTableData("refractive_index")
            assert n is not None
            # c / vg = n - lambda * (d_n/d_lambda)
            Nn, Ng = np.size(n), np.size(vg)
            lam_n = np.linspace(lamMin, lamMax, Nn)
            lam_g = np.linspace(lamMin, lamMax, Ng)
            dl = (lamMax - lamMin) / (Nn - 1)
            dn_dl = np.gradient(n, dl)
            vg_exp = speed_of_light / (n - lam_n * dn_dl)
            vg_exp = np.interp(lam_g, lam_n, vg)
            assert np.allclose(vg, vg_exp, rtol=rtol, atol=atol), "Group velocity does not match refractive index!"
        if (mu_a := getTableData("absorption_coef")) is not None:
            assert np.min(mu_a) >= 0.0, "Absorption coefficient must be non-negative!"
        if (mu_s := getTableData("scattering_coef")) is not None:
            assert np.min(mu_s) >= 0.0, "Scattering coefficient must be non-negative!"
        if (beta := getTableData("phase_sampling")) is not None:
            assert "log_phase_function" in medium.properties, "Phase sampling is provided, but no phase function!"
            assert np.all((beta >= -1.0) & (beta <= 1.0)), "Domain of phase sampling must be in [-1, 1]!"
        if (beta := getTableData("log_phase_function")) is not None:
            beta_s = getTableData("phase_sampling")
            assert beta_s is not None, "Phase function is defined, but no sampling!"
            N = np.size(beta)
            y = np.exp(beta)
            dx = 2.0 / (N - 1)
            cdf = 2 * np.pi * cumulative_simpson(y, dx=dx, initial=0)
            assert np.allclose(cdf[-1], 1.0, rtol=rtol, atol=atol), "Phase function must integrate over solid angles to 1!"
            u = np.linspace(-1, 1, N)
            quantiles = np.interp(beta_s, u, cdf)
            # sampling is fine in both directions, i.e. both u and 1-u are fine
            # this will only flip the sign in the difference -> use abs
            # if the sampling table is correct, the quantiles should be equidistant
            dq = np.abs(np.diff(quantiles))
            du = 1 / len(dq)
            assert np.allclose(dq, du, rtol=rtol, atol=atol), "Phase sampling function does not reproduce phase function!"
        if( m12 := getTableData("phase_m12")) is not None:
            assert np.all((m12 >= -1.0) & (m12 <= 1.0)), "Mueller matrix must be normalized!"
        if (m22 := getTableData("phase_m22")) is not None:
            assert np.all((m22 >= -1.0) & (m22 <= 1.0)), "Mueller matrix must be normalized!"
        if (m33 := getTableData("phase_m33")) is not None:
            assert np.all((m33 >= -1.0) & (m33 <= 1.0)), "Mueller matrix must be normalized!"
        if (m34 := getTableData("phase_m34")) is not None:
            assert np.all((m34 >= -1.0) & (m34 <= 1.0)), "Mueller matrix must be normalized!"

    except AssertionError as err:
        if raises:
            raise err
        else:
            return False
    # No errors
    return True


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

    LIGHT_SOURCE = 0x04
    """
    Marks the material as light source. Supporting such material is optional for
    tracers.
    """

    NO_REFLECT_FWD = 0x08
    """
    Forbids tracer to reflect forward rays from the material, but still takes
    the reflectivity factor (i.e. Fresnel equation) into account.
    """

    NO_REFLECT_BWD = 0x10
    """
    Forbids tracer to reflect backward rays from the material, but still takes
    the reflectivity factor (i.e. Fresnel equation) into account.
    """

    NO_REFLECT = 0x18
    """
    Forbids tracer to reflect rays from the material, but still takes the
    reflectivity factor (i.e. Fresnel equation) into account.
    """

    NO_TRANSMIT_FWD = 0x20
    """
    Forbids tracer to transmit forward rays through this material, but still
    takes the reflectivity factor (i.e. Fresnel equation) into account.
    """

    NO_TRANSMIT_BWD = 0x40
    """
    Forbids tracer to transmit backward rays through this material, but still
    takes the reflectivity factor (i.e. Fresnel equation) into account.
    """

    NO_TRANSMIT = 0x60
    """
    Forbids tracer to transmit rays through this material, but still takes the
    reflectivity factor (i.e. Fresnel equation) into account.
    """

    VOLUME_BORDER = 0x80
    """
    Marks the boundary of a volume, where media changes but the rays neither
    reflect nor refract, i.e. keep straight. Can be used to model inhomogeneous
    media.
    """

    SKIP_MEDIA_MISMATCH_TEST = 0x100
    """
    Tells the tracer to skip the test checking whether the ray comes from the
    expected medium. Usually, if a material describes the boundary between e.g.
    air and water, the tracer will check whether a ray actually comes from air
    before transmitting it into water. Skipping this test may be usefull in
    creating material that border multiple media on one side like a generic
    absorber material, but may cause errors in the geometry or scene to go
    unnoticed.
    """


_materialFlagsMap = {
    "B": MaterialFlags.BLACK_BODY,
    "D": MaterialFlags.DETECTOR,
    "L": MaterialFlags.LIGHT_SOURCE,
    "R": MaterialFlags.NO_REFLECT,
    "Rbf": MaterialFlags.NO_REFLECT,
    "Rfb": MaterialFlags.NO_REFLECT,
    "Rb": MaterialFlags.NO_REFLECT_BWD,
    "Rf": MaterialFlags.NO_REFLECT_FWD,
    "T": MaterialFlags.NO_TRANSMIT,
    "Tbf": MaterialFlags.NO_TRANSMIT,
    "Tfb": MaterialFlags.NO_TRANSMIT,
    "Tb": MaterialFlags.NO_TRANSMIT_BWD,
    "Tf": MaterialFlags.NO_TRANSMIT_FWD,
    "V": MaterialFlags.VOLUME_BORDER,
    "*": MaterialFlags.SKIP_MEDIA_MISMATCH_TEST,
}


def parseMaterialFlags(flags: str) -> MaterialFlags:
    """
    Parses the given string where each capital letter represents a flag as
    listed below. The flags `R` and `T` can further be specialized to only apply
    to forward or backward rays by appending a lower case `f` or `b`
    respectively.

    Flags:
     - `B` : `BLACK_BODY`
     - `D` : `DETECTOR`
     - `L` : `LIGHT_SOURCE`
     - `R` : removes `NO_REFLECT`
     - `T` : removes `NO_TRANSMIT`
     - `V` : `VOLUME_BORDER`
     - `*` : `SKIP_MEDIA_MISMATCH_TEST`
    """

    # tokenize
    tokens = re.findall(r"[A-Z\*][a-z]*", flags)
    # parse tokens
    result = MaterialFlags.NO_REFLECT | MaterialFlags.NO_TRANSMIT
    for token in tokens:
        if token in _materialFlagsMap:
            result ^= _materialFlagsMap[token]
        else:
            raise ValueError(f"Unknown material flag '{token}'")
    # done
    return result


class MaterialFlagsParsingDescriptor:
    """Descriptor parsing material flags on demand"""

    # See: https://docs.python.org/3/library/dataclasses.html#descriptor-typed-fields

    def __set_name__(self, owner, name) -> None:
        self._name = "_" + name

    def __get__(self, obj, type) -> MaterialFlags:
        if obj is None:
            # dataclass probing for default value
            return MaterialFlags(0)
        return getattr(obj, self._name, MaterialFlags(0))

    def __set__(self, obj, value: MaterialFlags | str):
        if isinstance(value, str):
            value = parseMaterialFlags(value)
        setattr(obj, self._name, value)

    def __delete__(self, obj) -> None:
        setattr(obj, self._name, MaterialFlags(0))


@dataclass
class Material:
    """
    Holds information about the material of a geometry. In general a geometry
    separates space into an "inside" and an "outside" defined by the winding
    order of its triangles, where outwards facing triangles have their vertices
    defined counter-clockwise. Materials assign a `Medium` to each of them,
    alongside additional properties.

    Parameters
    ----------
    name: str
        Name of this material. Can later be used to fetch the index of the
        material in the corresponding table.
    inside: Medium | str | None
        Medium in the inside of a geometry. Can also be specified by its name,
        which will get resolved during baking/serialization.
        `None` defaults to vacuum.
    outside: Medium | str | None
        Medium in the outside of a geometry. Can also be specified by its name,
        which will get resolved during baking/serialization.
    flags: (MaterialFlags|str,MaterialFlags|str)|MaterialFlags|str, default=0
        `MaterialFlags` applied to the material. In a tuple the first element
        applies to the inward direction, the second to the outward one.
        Otherwise the flags are applied to both directions. See
        `parseMaterialFlags` for a description on how to specify material flags
        with a string.
    properties: Iterable[Property] | None, default=None
        Optional additional properties assigned to the material.
    """

    def __init__(
        self,
        name: str,
        inside: Medium | str | None,
        outside: Medium | str | None,
        *,
        flags: (
            tuple[MaterialFlags | str, MaterialFlags | str] | MaterialFlags | str
        ) = MaterialFlags(0),
        properties: Mapping[str, Property] | None = None,
    ) -> None:
        self.name = name
        self.inside = inside
        self.outside = outside
        if isinstance(flags, tuple):
            self.flagsInward = flags[0]
            self.flagsOutward = flags[1]
        else:
            self.flagsInward = flags
            self.flagsOutward = flags
        self.properties = {} if properties is None else dict(properties)

    name: str
    """Name of the material"""
    inside: Medium | str | None
    """
    Medium in the inside of the geometry. Either stored directly or referenced
    by its name. A value of `None` corresponds to vacuum.
    """
    outside: Medium | str | None
    """
    Medium in the outside of the geometry. Either stored directly or referenced
    by its name. A value of `None` corresponds to vacuum.
    """
    flagsOutward: MaterialFlagsParsingDescriptor = MaterialFlagsParsingDescriptor()
    """
    Material flags specifying the behavior of rays hitting the material from
    the `inside` medium
    """
    flagsInward: MaterialFlagsParsingDescriptor = MaterialFlagsParsingDescriptor()
    """
    Material flags specifying the behavior of rays hitting the material from
    the `inside` medium
    """
    properties: dict[str, Property] = field(default_factory=dict)
    """
    Optional additional properties associated with this material.
    """


def saveMaterials(
    file: str | Path | ZipFile | ZipPath,
    material: Iterable[Material],
    *,
    media: Iterable[Medium] = [],
) -> None:
    """
    Stores the given materials alongside the referenced media at the given path,
    which can either point to a file or directory. The former case is equivalent
    to the latter, but compressed into a zip file.

    Parameters
    ----------
    path: str | bytes | PathLike
        Where to store the materials
    material: Iterable[Material]
        List of materials to store
    media: Iterable[Medium] = []
        List of extra media to store. Also used to lookup media referenced in
        materials by name.
    """
    # massage things until we end up with a ZipPath to make things simpler
    if isinstance(file, str):
        file = Path(file)
    if isinstance(file, Path):
        with ZipFile(file, "w") as zipFile:
            return saveMaterials(zipFile, material, media=media)
    if isinstance(file, ZipFile):
        file = ZipPath(file)

    # collect all media
    media_dict = {m.name: m for m in media}
    for mat in material:
        if isinstance(mat.inside, Medium) and mat.inside.name not in media_dict:
            media_dict[mat.inside.name] = mat.inside
        if isinstance(mat.outside, Medium) and mat.outside.name not in media_dict:
            media_dict[mat.outside.name] = mat.outside
    for mat in material:
        if isinstance(mat.inside, str) and mat.inside not in media_dict:
            warnings.warn(
                f'Material "{mat.name}" has unknown inside medium "{mat.inside}"'
            )
        if isinstance(mat.outside, str) and mat.outside not in media_dict:
            warnings.warn(
                f'Material "{mat.name}" has unknown outside medium "{mat.outside}"'
            )

    # save all media
    map = _createTableEntryFromMedium
    media_props = {name: map(m) for name, m in media_dict.items()}
    saveTable(file.joinpath("media/"), media_props)

    # save all material
    file = file.joinpath("material/")
    getName = lambda m: m.name if isinstance(m, Medium) else m
    printName = lambda m: m if (m := getName(m)) is not None else ""
    for mat in material:
        mat_path = file.joinpath(f"{mat.name}/")
        # write material description
        with mat_path.joinpath("material").open("w") as mat_file:
            name_inside = printName(mat.inside)
            name_outside = printName(mat.outside)
            mat_file.write(f"{str(int(mat.flagsInward))},{name_inside}\n")
            mat_file.write(f"{str(int(mat.flagsOutward))},{name_outside}\n")
        # optionally, write additional properties
        saveTableEntry(mat_path.joinpath("properties/"), mat.properties)


def loadMaterials(
    path: str | Path | ZipFile | ZipPath,
) -> tuple[dict[str, Material], dict[str, Medium]]:
    """
    Loads and returns the materials and media from the given path.

    Parameters
    ----------
    path: str | bytes | PathLike
        Where to load the materials and media from

    Returns
    -------
    materials: dict[str, Material]
        Dictionary indexing all loaded materials by their names
    media: dict[str, Medium]
        Dictionary indexing all loaded media by their names
    """
    # massage things until we end up with a ZipPath
    if isinstance(path, str):
        path = Path(path)
    if isinstance(path, Path):
        with ZipFile(path, "r") as zipFile:
            return loadMaterials(zipFile)
    if isinstance(path, ZipFile):
        path = ZipPath(path)

    # load media
    def convert(name: str, media: PropertyTableEntry) -> Medium:
        properties = dict(media)
        wavelength_range = properties.pop("wavelength_range")
        if not isinstance(wavelength_range, FloatProperty):
            raise ValueError(
                f'Property "wavelength_range" of media "{name}" has wrong type!'
            )
        return Medium(name, wavelength_range.value, properties)

    mediaTable = loadTable(path.joinpath("media/"))
    media_dict = {name: convert(name, med) for name, med in mediaTable.items()}

    # load material
    def parseLine(line: str):
        flags, med = line.split(",", 1)
        flags = MaterialFlags(int(flags))
        med = med[:-1] or None  # remove new line and replace empty with None
        if med in media_dict:
            med = media_dict[med]
        return (flags, med)

    mat_dict = {}
    for mat_path in path.joinpath("material/").iterdir():
        if not mat_path.is_dir():
            raise ValueError(f'Unexpected file "{mat_path}"')
        # load material description file
        with mat_path.joinpath("material").open("r") as file:
            flagsIn, medIn = parseLine(file.readline())
            flagsOut, medOut = parseLine(file.readline())
        name = mat_path.name
        flags = (flagsIn, flagsOut)
        # load optional additional properties
        props = loadTableEntry(mat_path.joinpath("properties/"))
        mat_dict[name] = Material(name, medIn, medOut, flags=flags, properties=props)

    # done
    return (mat_dict, media_dict)


def getPropertySamples(src: Medium | Material, name: str) -> NDArray | None:
    """
    Fetches the property specified by name from the given medium or material
    and returns the corresponding samples if it is a `TableProperty`. If the
    property is not present or of different type, returns `None`.
    """
    prop = src.properties[name]
    if isinstance(prop, TableProperty) and prop.table is not None:
        return prop.table.samples
    return None


VACUUM_IDX: Final[int] = 0xFFFFFFFF
"""Special value for indexing the medium table indicating vacuum"""


class MaterialInterfaceProperty(Property, ext="mat"):
    """
    Combines material flags and the index of the medium on the other side.
    Stores them internally as a tuple of two 32 bit integers, the first being
    the index of the medium in the media table and the other the material flags.
    """

    def __init__(self, mediumIdx: int | None, flags: MaterialFlags) -> None:
        super().__init__()
        self.mediumIdx = mediumIdx
        self.flags = flags

    @property
    def flags(self) -> MaterialFlags:
        """Flags specifying the behavior of the tracer encountering this surface"""
        # only use self.data as single source of truth
        return MaterialFlags(unpack("<II", pack("<Q", self.data))[1])

    @flags.setter
    def flags(self, value: MaterialFlags) -> None:
        idx, _ = unpack("<II", pack("<Q", self.data))  # property may return None
        self.data = unpack("<Q", pack("<II", idx, int(value)))[0]

    @property
    def mediumIdx(self) -> int | None:
        """Index in the media table of the medium on the other side"""
        # only use self.data as single source of truth
        idx = unpack("<II", pack("<Q", self.data))[0]
        return idx if idx != VACUUM_IDX else None

    @mediumIdx.setter
    def mediumIdx(self, value: int | None) -> None:
        if value is None:
            value = VACUUM_IDX  # use max value to indicate None
        self.data = unpack("<Q", pack("<II", value, self.flags))[0]


def _createTableEntryFromMaterial(
    material: Material,
    mediaTable: PropertyTable,
) -> PropertyTableEntry:
    """Util function converting material to a property table entry"""
    # fetch media names
    getMedium = lambda m: m.name if isinstance(m, Medium) else m
    inside: str | None = getMedium(material.inside)
    outside: str | None = getMedium(material.outside)
    # check if media is present
    for m in (inside, outside):
        if m is not None and m not in mediaTable:
            raise ValueError(f"Material {material.name} references unknown medium {m}")
    # look up media indices
    getIdx = lambda m: m if m is None else mediaTable[m]
    props: dict[str, Property] = {
        "inwards": MaterialInterfaceProperty(getIdx(inside), material.flagsInward),
        "outwards": MaterialInterfaceProperty(getIdx(outside), material.flagsOutward),
    }
    # add additional material properties
    props.update(material.properties)
    return props


DefaultMediaSlots: Final[list[str]]
"""Order of default media properties used by material store"""
DefaultMediaSlots = [
    "wavelength_range",
    OpticalProperties.REFRACTIVE_INDEX,
    OpticalProperties.GROUP_VELOCITY,
    OpticalProperties.ABSORPTION_COEF,
    OpticalProperties.SCATTERING_COEF,
    OpticalProperties.LOG_PHASE_FUNC,
    OpticalProperties.PHASE_SAMPLING,
    OpticalProperties.PHASE_M12,
    OpticalProperties.PHASE_M22,
    OpticalProperties.PHASE_M33,
    OpticalProperties.PHASE_M34,
]

DefaultMaterialSlots: Final[list[str]] = ["inwards", "outwards"]
"""Order of default material properties used by material store"""


class MaterialStore:
    """
    Manages the upload and lifetime of media and material on the device.

    Parameters
    ----------
    material: Iterable[Material]
        Materials to be uploaded to the GPU
    media: Iterable[Medium], default=[]
        Additional media to be uploaded to the GPU. Media that are referenced by
        materials directly (i.e. not by their name, but stored as variable) will
        be included automatically and do not need to be listed here again.
    mediaSlots: Iterable[str] | None, default=DefaultMediaSlots
        List of property slots to be defined in the shader. If any media defines
        additional properties, these will be added to the end of list. If
        `None`, only the ones specified by the media will be included and
        certain ones might be missing if none of the media specify them. Order
        of the specified slots will be preserved. This can thus be used to force
        a specific order to make compiled shader code compatible.
    materialSlots: Iterable[str] | None, default=DefaultMaterialSlots
        Same as `mediaSlots`, but used for material properties.
    """

    def __init__(
        self,
        material: Iterable[Material],
        *,
        media: Iterable[Medium] = [],
        mediaSlots: Iterable[str] | None = DefaultMediaSlots,
        materialSlots: Iterable[str] | None = DefaultMaterialSlots,
    ) -> None:
        # collect all media
        mediaDict = {m.name: _createTableEntryFromMedium(m) for m in media}
        for med in chain.from_iterable((m.inside, m.outside) for m in material):
            if isinstance(med, Medium) and med.name not in mediaDict:
                mediaDict[med.name] = _createTableEntryFromMedium(med)
        # create media table
        self._mediaTable = PropertyTable(mediaDict, requiredSlots=mediaSlots)

        # create all materials
        _map = _createTableEntryFromMaterial
        materialDict = {m.name: _map(m, self._mediaTable) for m in material}
        self._materialTable = PropertyTable(materialDict, requiredSlots=materialSlots)

    @property
    def header(self) -> dict[str, str]:
        """Creates the header defining the slots as expected by the shader code"""
        return {"slots.glsl": createPreamble(**self.slotMacros)}

    @property
    def media(self) -> PropertyTable:
        """Table containing the media properties"""
        return self._mediaTable

    @property
    def materials(self) -> PropertyTable:
        """Table containting the material properties"""
        return self._materialTable

    @property
    def slotMacros(self) -> dict[str, int]:
        """Dictionary containing the slot names to be used in shaders"""
        mediaSlots = createSlotMacros(self.media, "MEDIA_SLOT")
        materialSlots = createSlotMacros(self.materials, "MATERIAL_SLOT")
        return mediaSlots | materialSlots

    def bindParams(self, program: hp.Program) -> None:
        program.bindParams(
            MediaTable=self.media.tensor,
            MaterialTable=self.materials.tensor,
        )

    def __len__(self) -> int:
        return len(self.materials)

    def __iter__(self) -> Iterator[str]:
        return iter(self.materials)

    def __contains__(self, material: str) -> bool:
        """Checks whether a material of the given name exists in this store"""
        return material in self.materials

    def __getitem__(self, material: str) -> int:
        """Returns the index of the material with the given name."""
        return self.materials[material]


#################################### MODELS ####################################


_R = TypeVar("_R")


class _range_checked_call(Generic[_R]):
    """util class for performing range checks before passing values to function"""

    def __init__(self, fn, obj, range) -> None:
        self.fn = fn
        self.obj = obj
        self.range = range

    def __call__(self, *xi) -> _R:
        xi = [np.asarray(x) for x in xi]
        for i, (x, (lo, hi)) in enumerate(zip(xi, self.range)):
            if x.min() < lo or x.max() > hi:
                raise ValueError(
                    f"Values for parameter {i+1} outside the allowed range: {lo} - {hi}"
                )
        return self.fn(self.obj, *xi)


class _medium_property(Generic[_R]):
    """Class marking medium properties in models. See `medium_property`"""

    def __init__(
        self,
        fn: Callable[..., _R],
        *,
        name: str | None = None,
        range: list[tuple[float, float]] | tuple[float, float] | None = None,
        interpolation: Interpolation = "linear",
    ) -> None:
        self.fn = fn
        self.name = name
        if isinstance(range, tuple):
            range = [range]
        self.range = range
        self.interpolation: Interpolation = interpolation

    def __set_name__(self, owner, name: str) -> None:
        if self.name is None:
            self.name = name

    @overload
    def __get__(self, obj: None, owner=None) -> Self: ...

    @overload
    def __get__(self, obj: Any, owner=None) -> _range_checked_call[_R]: ...

    def __get__(self, obj, owner=None):
        if obj is None:
            # accessed from class
            return self
        range = [obj.wavelengthRange] if self.range is None else self.range
        return _range_checked_call(self.fn, obj, range)

    def createProperty(
        self,
        obj,
        defaultRange: tuple[float, float],
        numSamples: int,
        const: bool,
    ) -> Property:
        """Creates a `Property` from this medium property"""
        if const:
            value = self.fn(obj)
            return FloatProperty(value)
        else:
            range = [defaultRange] if self.range is None else self.range
            xi = [(*r, numSamples) for r in range]
            fn = lambda *xi: self.fn(obj, *xi)  # bind object as self
            table = createTableFromFunction(fn, *xi, interpolation=self.interpolation)
            return TableProperty(table)


# TODO: Fix these type annotations
@overload
def medium_property(
    fn: Callable[..., _R],
    *,
    name: None | str = None,
    range: list[tuple[float, float]] | tuple[float, float] | None = None,
    interpolation: Interpolation = "linear",
) -> _medium_property[_R]: ...


@overload
def medium_property(
    fn: None = None,
    *,
    name: None | str = None,
    range: list[tuple[float, float]] | tuple[float, float] | None = None,
    interpolation: Interpolation = "linear",
) -> Callable[[Callable[..., _R]], _medium_property[_R]]: ...


def medium_property(func=None, *, name=None, range=None, interpolation="linear"):
    """
    Can be used in classes derived from `MediumModel` to mark functions or
    properties as medium properties. These will be used during medium creation
    to populate the corresponding `PropertyTable`.

    Parameters
    ----------
    fn:
        Function to decorate
    name: str | None, default=None
        Name of the property. If `None`, uses the function's name
    range: ((float, float), ...) | None, default=None
        Range over which to sample the function for each axis. If `None`, uses
        the wavelength range of the model. Ignored on properties.
    interpolation: Interpolation, default="linear"
        Interpolation to use when creating look up table from the sampled
        values. Ignored on properties.

    Examples
    --------
    >>> class Model(MediumModel):
    ...     @property
    ...     @medium_property
    ...     def temperature(self) -> float:
    ...         return 21.0
    ...     @medium_property
    ...     def refractive_index(self, wavelength):
    ...         return 1.41 * np.ones_like(wavelength)
    ...     @medium_property(range=(-1.,1.))
    ...     def log_phase_function(self, cos_theta):
    ...         return np.ones_like(cos_theta) / (4.0 * np.pi)
    """
    # need this trick to allow arguments
    if func is not None:
        # decorated without parameters -> name is actually the function
        return _medium_property(func)
    else:
        return lambda fn: _medium_property(
            fn, name=name, range=range, interpolation=interpolation
        )


class MediumModel:
    """
    Base class for medium models. Implements a function for creating a medium
    by collecting and where necessary sampling properties defined in derived
    classes. Properties can be defined using the `mediumProperty` decorator.

    Parameters
    ----------
    name: str, default="noname"
        Name of the model. Will be used as default name for created media.
    wavelengthRange: (float, float), default=(100.0, 1500.0) nm
        Range of wavelength in nm for which the model is valid. Sampling of
        optical properties will use this range where applicable.
    """

    def __init__(
        self,
        *args,
        name: str = "noname",
        wavelengthRange: tuple[float, float] = (100.0, 1500.0) * u.nm,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.wavelengthRange = wavelengthRange

    @classproperty
    def propertyNames(cls) -> set[str]:
        """Name of all medium properties defined by this model"""
        names = set()
        for atr in dir(cls):
            if atr.startswith("_"):
                continue
            if atr == "propertyNames":
                continue  # prevent infinite recursion
            prop = getattr(cls, atr)
            if isinstance(prop, property):
                prop = prop.fget
            if isinstance(prop, _medium_property):
                names.add(prop.name)
        return names

    @property
    def wavelengthRange(self) -> tuple[float, float]:
        """Wavelength range for which this model is valid"""
        return self._wavelengthRange

    @wavelengthRange.setter
    def wavelengthRange(self, value: tuple[float, float]) -> None:
        lo, hi = value
        # check using not to propertly handle NaN values
        if not lo <= hi:
            raise ValueError("Wavelength range is not ordered!")
        if not lo >= 0.0:
            raise ValueError("Wavelength range must be positive!")
        if not hi < float("inf"):
            raise ValueError("Wavelength range must be finite!")
        self._wavelengthRange = value

    def restrictWavelengthRange(self, range: tuple[float, float]) -> None:
        """
        Restricts the current wavelength range by the given one, that is the new
        range will be the intersection of both. Raises an error if the
        intersection is empty.
        """
        self.wavelengthRange = intersectRange(self.wavelengthRange, range)

    def createMedium(
        self,
        *,
        wavelengthRange: tuple[float, float] | None = None,
        numSamples: int = 1024,
        name: str | None = None,
    ) -> Medium:
        if wavelengthRange is None:
            wavelengthRange = self.wavelengthRange
        if name is None:
            name = self.name

        # collect medium properties
        props = {}
        for atr in dir(self):
            if atr.startswith("_"):
                continue
            # fetch properties from class object to sidestep descriptor
            prop = getattr(self.__class__, atr, None)
            if prop is None:
                continue
            const = isinstance(prop, property)
            if const:
                prop = prop.fget
            if isinstance(prop, _medium_property):
                props[prop.name] = prop.createProperty(
                    self, wavelengthRange, numSamples, const
                )

        return Medium(name, wavelengthRange, props)


class NumericalPhaseSamplingMixin(MediumModel):
    """
    Mixin class providing a phase sampling function by numerical inversion of
    either the `phase_function` or `log_phase_function` method. The former takes
    precedence. Optionally caches the underlying sampler instance. The cache can
    be invalidated via a call to `_invalidatePhaseSampler`.

    Parameters
    ----------
    cache: bool, default=True
        Whether to cache the phase sampler.
    """

    class _DistWrapper:
        """Wraps `phase_function` method for use with scipy"""

        def __init__(self, model) -> None:
            self._model = model

        def pdf(self, x: float) -> float:
            return self._model.phase_function(x).item()

    class _LogDistWrapper:
        """Wraps `log_phase_function` method for use with scipy"""

        def __init__(self, model) -> None:
            self._model = model

        def logpdf(self, x: float) -> float:
            return self._model.log_phase_function(x).item()

    def __init__(self, *args, cachePhaseSampler: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cachePhaseSampler = cachePhaseSampler
        self._phaseSampler = None

    def _invalidatePhaseSampler(self) -> None:
        """Marks the cached phase sampler as invalid"""
        self._phaseSampler = None

    @medium_property(range=(0.0, 1.0))
    def phase_sampling(self, eta: NDArray) -> NDArray:
        if self._phaseSampler is not None:
            sampler = self._phaseSampler
        else:
            if "phase_function" in dir(self):
                dist = NumericalPhaseSamplingMixin._DistWrapper(self)
            elif "log_phase_function" in dir(self):
                dist = NumericalPhaseSamplingMixin._LogDistWrapper(self)
            else:
                raise NotImplementedError("Model provides no phase function!")
            sampler = NumericalInversePolynomial(dist, domain=(-1, 1))
            if self._cachePhaseSampler:
                self._phaseSampler = sampler
        return sampler.ppf(eta)


class SellmeierEquation(MediumModel):
    """
    The Sellmeier equation is a empirical model of the refractive index as a
    function of the wavelength parameterized by a set of 6 constants for some
    transparent media.

    .. math::
        n^2(\\lambda) = 1 + \\sum_i\\frac{B_i\\lambda^2}{\\lambda^2-C_i}

    See https://en.wikipedia.org/wiki/Sellmeier_equation
    """

    def __init__(
        self,
        B1: float,
        B2: float,
        B3: float,
        C1: float,
        C2: float,
        C3: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3

    @medium_property
    def refractive_index(self, wavelength: NDArray) -> NDArray:
        """Calculates the refractive index for the given wavelengths"""
        L2 = np.square(wavelength)
        # L2 = np.square(wavelength)
        S1 = self.B1 * L2 / (L2 - self.C1)
        S2 = self.B2 * L2 / (L2 - self.C2)
        S3 = self.B3 * L2 / (L2 - self.C3)
        return np.sqrt(1.0 + S1 + S2 + S3)

    @medium_property
    def group_velocity(self, wavelength: NDArray) -> NDArray:
        """
        Calculates the group velocity in for the given wavelengths
        """
        L = wavelength
        n = self.refractive_index(L)
        L2 = np.square(L)
        S1 = self.B1 * self.C1 * L / np.square(L2 - self.C1)
        S2 = self.B2 * self.C2 * L / np.square(L2 - self.C2)
        S3 = self.B3 * self.C3 * L / np.square(L2 - self.C3)
        grad = -1.0 * (S1 + S2 + S3) / n
        return 1.0 / (n - L * grad) * u.c


class BK7Model(SellmeierEquation):
    """
    Model for BK7 glass based on the Sellmeier equation for refractive index
    and measurements for the absorption coefficient.

    See https://www.schott.com/shop/advanced-optics/en/Optical-Glass/SCHOTT-N-BK7/c/glass-SCHOTT%20N-BK7%C2%AE
    """

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
            name="bk7",
        )
        self.restrictWavelengthRange((200.0, 1060.0) * u.nm)
        # lazily load data
        if BK7Model.TransmissionTable is None:
            BK7Model.TransmissionTable = loadCSV("bk7_transmission.csv", skiprows=2)

    @medium_property
    def absorption_coef(self, wavelength: NDArray) -> NDArray:
        """Returns the absorption coefficient for the given wavelengths"""
        assert BK7Model.TransmissionTable is not None
        # we can transform the transmission measurements to absorption
        # coefficients via the Beer-Lambert law. Unfortunately, they two
        # measurements disagree a lot. So we take the average weighted by
        # the probe thickness, as thicker ones should give a better result
        # To avoid taking the average with inf, we actually take the average
        # of the absorption lengths, i.e. inf -> 0

        # disable error, since we'll take the log of zero
        with np.errstate(divide="ignore"):
            tau_10mm = -0.010 / np.log(BK7Model.TransmissionTable[:, 1])
            tau_25mm = -0.025 / np.log(BK7Model.TransmissionTable[:, 2])
            tau_avg = (10.0 * tau_10mm + 25.0 * tau_25mm) / 35.0
            # interpolate without inf
            tau = np.interp(
                u.convert(wavelength, u.nm), BK7Model.TransmissionTable[:, 0], tau_avg
            )
            # convert back to coefficients
            return np.reciprocal(tau) / u.m


class RayleighScatteringPhaseFunction(NumericalPhaseSamplingMixin, MediumModel):
    """
    Models the phase function and Mueller matrix elements according to Rayleigh
    scattering as shown in [ZH21] and [ZH19]. Lacks the model for the scattering
    coefficient, which can be found in `RayleighScatteringModel`.

    Parameters
    ----------
    depolarizationRatio: float, default=0.0
        Depolarization ratio of the scattering medium, defined as the ratio
        between horizontally and vertically polarized components of the
        scattered light at 90 degrees from an unpolarized incident light source.

    References
    ----------
    [ZH21] X. Zhang, and L. Hi, "Light Scattering by Pure Water and Seawater: Recent Development", Journal of Remote Sensing, 2021
    [ZH19] X. Zhang, et al., "Light scattering py pure water and seawater: the depolarization ratio and its variation with salinity", Applied Optics Vol. 58 No. 4, 2019
    """

    def __init__(self, *args, depolarizationRatio: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.depolarizationRatio = depolarizationRatio

    @medium_property(range=(-1.0, 1.0))
    def phase_function(self, cos_theta: NDArray) -> NDArray:
        #                    1 - delta
        # beta(theta) ~ 1 + ----------- cos^2(theta)
        #                    1 + delta
        delta = self.depolarizationRatio
        phase = 1 + (1 - delta) / (1 + delta) * cos_theta**2
        I = 8 * np.pi / 3 * (2 + delta) / (1 + delta)  # integral phase from -1 to 1
        return phase / I

    @medium_property(range=(-1.0, 1.0))
    def log_phase_function(self, cos_theta: NDArray) -> NDArray:
        return np.log(self.phase_function(cos_theta))

    @medium_property(range=(-1.0, 1.0))
    def phase_m12(self, cos_theta: NDArray) -> NDArray:
        # M_11 = beta_90 * 1 + (1-d)/(1+d)cos^2
        # M_12 = beta_90 * -(1-d)/(1+d)sin^2
        #            M_12       -(1-d) sin^2
        # -> m_12 = ------ = ------------------
        #            M_11     1+d + (1-d)cos^2
        d = self.depolarizationRatio
        u = cos_theta
        return -(1 - d) * (1 - u**2) / (1 + d + (1 - d) * u**2)

    @medium_property(range=(-1.0, 1.0))
    def phase_m22(self, cos_theta: NDArray) -> NDArray:
        # M_22 = M_11 -> m_22 = 1
        return np.ones_like(cos_theta)

    @medium_property(range=(-1.0, 1.0))
    def phase_m33(self, cos_theta: NDArray) -> NDArray:
        # M_11 = beta_90 * 1 + (1-d)/(1+d)cos^2
        # M_33 = beta_90 * 2(1-d)/(1+d)cos
        #            M_33        2(1-d)cos
        # -> m_33 = ------ = ------------------
        #            M_11     1+d + (1-d)cos^2
        d = self.depolarizationRatio
        u = cos_theta
        return 2 * (1 - d) * u / (1 + d + (1 - d) * u**2)


class RayleighScatteringModel(RayleighScatteringPhaseFunction):
    """
    Models the typical scattering coefficient used with Rayleigh scattering
    caused by thermodynamic fluctuations following the Einstein-Smoluchowski
    theory including correction factor for the depolarization ratio.

    Parameters
    ----------
    temperature: float [°C]
        Temperature of the scattering medium
    betaT: float [m^3/J]
        Isothermal compressibility of the media
    depolarizationRatio: float
        Ratio of horizontally to vertically polarized light scattered at 90
        degrees for unpolarized incident light

    See
    ---
    RayleighScatteringPhaseFunction
    """

    def __init__(
        self,
        *args,
        temperature: float,
        betaT: float,
        depolarizationRatio: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, depolarizationRatio=depolarizationRatio, **kwargs)
        self.temperature = temperature
        self.betaT = betaT

    @medium_property
    def scattering_coef(self, wavelength: NDArray) -> NDArray:
        #      4pi^3 * k_B*T * beta_T                  2 + delta
        # b = ------------------------ * LL^2 * f_C * -----------
        #            2 * lam^4                         1 + delta
        lam = u.convert(wavelength, u.m)
        T = self.temperature
        betaT = self.betaT
        delta = self.depolarizationRatio
        E = scipy.constants.Boltzmann * (T + 273.15)
        # Cabannes factor; See [ZH21] Eq. 4
        fc = (6 + 6 * delta) / (6 - 7 * delta)
        fc *= (2 + delta) / (1 + delta)  # from integration
        # Lorentz-Lorenz; See [ZH21] Eq. 9
        try:
            n: NDArray = self.refractive_index(wavelength)
        except:
            raise RuntimeError(
                "Rayleigh scattering requires refractive index to be defined!"
            )
        n2 = n**2
        ll = (n2 - 1) * (1 + 2 / 3 * (n2 + 2.0) * ((n2 - 1) / (3 * n)) ** 2)

        return 4 * np.pi**3 / 3 * E * betaT / lam**4 * ll**2 * fc


class HenyeyGreensteinPhaseFunction(MediumModel):
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

    def __init__(self, g: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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

    @medium_property(range=(-1.0, 1.0))
    def log_phase_function(self, cos_theta: NDArray) -> NDArray:
        """
        Evaluates the log phase function for the given angles as cos(theta).
        Normalized with respect to unit sphere.
        """
        return np.log(
            (1.0 - self.g**2)
            / np.power(1.0 + self.g**2 - 2 * self.g * cos_theta, 1.5)
            / (4.0 * np.pi)
        )

    @medium_property(range=(0.0, 1.0))
    def phase_sampling(self, eta: NDArray) -> NDArray:
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


class FournierForandPhaseFunction(NumericalPhaseSamplingMixin):
    """
    Approximation of scattering in an ensemble of spherical particles with an
    refractive index n following a hyperbolic (Junge) size distribution
    N(r) ~ r^(-mu) where r is the radius and mu the slope of log N(r).

    See Fournier, G. and M. Jonasz, 1999. Computer based underwater imaging
        analysis. In Airborne and In-water Underwater Imaging,
        SPIE Vol. 3761, G. Gilbert [ed], 62-77 (with corrections)
    """

    def __init__(self, n: float, mu: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._n = n
        self._mu = mu

    @property
    def n(self) -> float:
        """The refractive index of the particles"""
        return self._n

    @n.setter
    def n(self, value: float) -> None:
        self._n = value
        self._invalidatePhaseSampler()

    @property
    def mu(self) -> float:
        """The slope of the particle size distribution log N(r) ~ -mu*log(r)"""
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        self._mu = value
        self._invalidatePhaseSampler()

    @medium_property(range=(-1.0, 1.0))
    def log_phase_function(self, cos_theta: ArrayLike) -> NDArray:
        """Evaluates the log phase function for the given angles mu = cos(theta)"""
        # phase functions becomes singular at cos_theta = 1.0
        # clip close before (1 float ulp)
        cos_theta = np.clip(cos_theta, -1.0, 1.0 - 1e-5)
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


class DispersionFreeMedium(MediumModel):
    """
    Idealized medium showing constant optical properties regardless of
    wavelength. Primarily intended for debugging purposes.

    Parameters
    ----------
    n: float, default=1.0
        Refractive index.
    ng: float | None, default=None
        Group index (group velocity = c/ng). If `None`, same as `n`.
    mu_a: float, default=0.0
        Absorption coefficient.
    mu_s: float, default=0.0
        Scattering coefficient.
    name: str, default="constant"
        Name of the model
    """

    def __init__(
        self,
        *args,
        n: float = 1.0,
        ng: float | None = None,
        mu_a: float = 0.0,
        mu_s: float = 0.0,
        name: str = "constant",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            name=name,
            **kwargs,
        )
        self.n = n
        self.ng = n if ng is None else ng
        self.mu_a = mu_a
        self.mu_s = mu_s

    @medium_property
    def refractive_index(self, wavelength: ArrayLike) -> NDArray:
        if not np.min(wavelength) >= 0.0:
            raise ValueError("wavelength must be positive!")
        return np.ones_like(wavelength) * self.n

    @medium_property
    def group_velocity(self, wavelength: ArrayLike) -> NDArray:
        if not np.min(wavelength) >= 0.0:
            raise ValueError("wavelength must be positive!")
        return np.ones_like(wavelength) / self.ng * u.c

    @medium_property
    def absorption_coef(self, wavelength: ArrayLike) -> NDArray:
        if not np.min(wavelength) >= 0.0:
            raise ValueError("wavelength must be positive!")
        return np.ones_like(wavelength) * self.mu_a

    @medium_property
    def scattering_coef(self, wavelength: ArrayLike) -> NDArray:
        if not np.min(wavelength) >= 0.0:
            raise ValueError("wavelength must be positive!")
        return np.ones_like(wavelength) * self.mu_s


class WaterBaseModel(MediumModel):
    """
    Models refractive index and group velocity of pure (sea) water only
    implementing the algorithm presented in [MS90]

    Parameters
    ----------
    temperature: float [°C]
        Water temperature. Must be in the range [-12, 80] °C
    pressure: float [dbar]
        Water pressure in dbar. Must be in the range [-0.5, 10000] dbar
    salinity: float [g/kg]
        Water salinity in gram per kilogram. Must be in the range [0, 120] g/kg

    Note
    ----
    [MS90] only claims correct results over a much smaller range, but it is
    still reasonable correct for values outside that range (See e.g. [ZH21]
    fig. 4)

    References
    ----------
    [MS90] R. C. Millard, and G. Seaer, "An index of refraction algorithm for
           seawater over temperature, pressure, salinity, density and
           wavelength", Deep Sea Research Vol. 37, 1909-1926 (1990)
    [ZH21] X. Zhang, L. Hu, "Light Scattering by Pure Water and Seawater:
           Recent Development", J Remote Sens. 2021, (2021)
    """

    # shared constants for Millard Seaer model
    _MS_CONSTS = SimpleNamespace(
        A0=1.3280657,
        L2=-0.0045536802,
        LM2=0.0025471707,
        LM4=0.000007501966,
        LM6=0.000002802632,
        T1=-0.0000052883907,
        T2=-0.0000030738272,
        T3=0.000000030124687,
        T4=-2.0863178e-10,
        TL=0.000010508621,
        T2L=0.00000021282248,
        T3L=-0.000000001705881,
        S0=0.00019029121,
        S1LM2=0.0000024239607,
        S1T=-0.00000073960297,
        S1T2=0.0000000089818478,
        S1T3=1.2078804e-10,
        STL=-0.0000003589495,
        P1=0.0000015868363,
        P2=-1.574074e-11,
        PLM2=0.000000010712063,
        PT=-0.0000000094634486,
        PT2=1.0100326e-10,
        P2T2=5.8085198e-15,
        P1S=-0.0000000011177517,
        PTS=5.7311268e-11,
        PT2S=-1.5460458e-12,
    )

    def __init__(
        self,
        temperature: float,
        pressure: float,
        salinity: float,
        *args,
        name: str = "water",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            name=name,
            **kwargs,
        )
        self.restrictWavelengthRange((250.0, 1100.0) * u.nm)
        # set props as they will do the range check
        self.temperature = temperature
        self.pressure = pressure
        self.salinity = salinity

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if not -12 <= value <= 80:
            raise ValueError("Temperature must be in the range [-12,80]°C!")
        self._temperature = value

    @property
    def pressure(self) -> float:
        """Water pressure in dbar"""
        return self._pressure

    @pressure.setter
    def pressure(self, value: float) -> None:
        if not -0.5 <= value <= 10000:
            raise ValueError("Pressure must be in the range [-0.5,10000]dbar!")
        self._pressure = value

    @property
    def salinity(self) -> float:
        """Water salinity in g/kg"""
        return self._salinity

    @salinity.setter
    def salinity(self, value: float) -> None:
        if not 0 <= value <= 120:
            raise ValueError("Salinity must be in the range [0,120]g/kg!")
        self._salinity = value

    @medium_property
    def refractive_index(self, wavelength: NDArray) -> NDArray:
        # formula expects wavelengths in micrometers -> convert
        L = u.convert(wavelength, u.um)
        T = self.temperature
        p = self.pressure
        S = self.salinity / (35.16504 / 35)  # convert ppt -> PSU
        C = self._MS_CONSTS
        # following naming scheme from paper
        N1 = (
            C.A0
            + C.L2 * (L**2)
            + C.LM2 / (L**2)
            + C.LM4 / (L**4)
            + C.LM6 / (L**6)
            + C.T1 * T
            + C.T2 * (T**2)
            + C.T3 * (T**3)
            + C.T4 * (T**4)
            + C.TL * T * L
            + C.T2L * (T**2) * L
            + C.T3L * (T**3) * L
        )
        N2 = (
            C.S0 * S
            + C.S1LM2 * S / (L**2)
            + C.S1T * S * T
            + C.S1T2 * S * (T**2)
            + C.S1T3 * S * (T**3)
            + C.STL * S * T * L
        )
        N3 = (
            C.P1 * p
            + C.P2 * (p**2)
            + C.PLM2 * p / (L**2)
            + C.PT * p * T
            + C.PT2 * p * (T**2)
            + C.P2T2 * (p**2) * (T**2)
        )
        N4 = C.P1S * p * S + C.PTS * p * T * S + C.PT2S * p * (T**2) * S
        return N1 + N2 + N3 + N4

    @medium_property
    def group_velocity(self, wavelength: NDArray) -> NDArray:
        # formula expects wavelengths in micrometers -> convert
        L = u.convert(wavelength, u.um)
        T = self.temperature
        p = self.pressure
        S = self.salinity / (35.16504 / 35)  # convert ppt -> PSU
        C = self._MS_CONSTS
        # G_i = dN_i/dL
        G1 = (
            2.0 * C.L2 * L
            - 2.0 * C.LM2 / (L**3)
            - 4.0 * C.LM4 / (L**5)
            - 6.0 * C.LM6 / (L**7)
            + C.TL * T
            + C.T2L * (T**2)
            + C.T3L * (T**3)
        )
        G2 = -2.0 * C.S1LM2 * S / (L**3) + C.STL * S * T
        G3 = -2.0 * C.PLM2 * p / (L**3)
        G4 = 0.0
        G = G1 + G2 + G3 + G4
        # vg = c / (n - L*dn/dL)
        n = self.refractive_index(wavelength)
        return 1.0 / (n - L * G) * u.c


class PureWaterModel(WaterBaseModel, RayleighScatteringPhaseFunction):
    """
    Model of the optical properties of pure (sea) water combining several
    numerical models and measurements: Refractive index and scattering
    coefficient are implemented according to [MS90] and [ZH21] respectively.
    Scattering distribution is assumed to follow Rayleigh. Absorption
    coefficients are interpolated using the data from [MCF16] in the range of
    (250-550)nm and [BHD94] for (550-800)nm. Correction for the temperature and
    salinity dependency of the absorption are not included.

    Parameters
    ----------
    temperature: float [°C]
        Water temperature. Must be in the range [-12, 80] °C
    pressure: float, [dbar]
        Water pressure in dbar. Must be in the range [-0.5, 10'000] dbar
    salinity: float, [g/kg]
        Water salinity in gram per kilogram. Must be in the range [0, 120] g/kg

    Note
    ----
    Optical properties of natural waters are heavily dependent on their
    constituents and thus are likely to be poorly modeled by this model.

    References
    ----------
    [BHD94] H. Buiteveld and J. M. H. Hakvoort and M. Donze, "The optical
            properties of pure water," in SPIE Proceedings on Ocean Optics XII,
            edited by J. S. Jaffe, 2258, 174--183, (1994).
    [MCF16] John D. Mason, Michael T. Cone, and Edward S. Fry, "Ultraviolet
            (250-550 nm) absorption spectrum of pure water," Appl. Opt. 55,
            7163-7172 (2016)
    [MS90] R. C. Millard, and G. Seaer, "An index of refraction algorithm for
           seawater over temperature, pressure, salinity, density and
           wavelength", Deep Sea Research Vol. 37, 1909-1926 (1990)
    [ZH21] X. Zhang, L. Hu, "Light Scattering by Pure Water and Seawater:
           Recent Development", J Remote Sens. 2021, (2021)

    See
    ---
    RayleighScatteringPhaseFunction
    """

    # lazily load data tables
    _MU_A_SPLINE = None

    def __init__(
        self,
        *args,
        temperature: float = 15.0,
        pressure: float = 0.0,
        salinity: float = 0.0,
        name: str = "water",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            temperature=temperature,
            pressure=pressure,
            salinity=salinity,
            depolarizationRatio=0.039,
            name=name,
            **kwargs,
        )
        self.restrictWavelengthRange((250.0, 800.0) * u.nm)
        # lazily load interpolation table
        if PureWaterModel._MU_A_SPLINE is None:
            mcf = loadCSV("water_mcf16.csv")
            bhd = loadCSV("water_bhd94.csv")
            lam_cut = mcf[-1, 0]
            bhd_cut = bhd[bhd[:, 0] > lam_cut]
            table = np.concatenate((mcf, bhd_cut), 0)
            PureWaterModel._MU_A_SPLINE = CubicSpline(table[:, 0], table[:, 1])

    @medium_property
    def absorption_coef(self, wavelength: NDArray) -> NDArray:
        # linear interpolate table
        assert PureWaterModel._MU_A_SPLINE is not None  # make the linter happy...
        return PureWaterModel._MU_A_SPLINE(wavelength)

    def _dg_dp(self) -> float:
        """partial derivative of Gibbs function w.r.t pressure"""
        # Code based on matlab code from [ZH21]
        x2 = 0.0248826675584615 * self.salinity
        x = np.sqrt(x2)
        y = self.temperature * 0.025
        z = self.pressure * 1e-4

        # fmt: off
        g03 = 100015.695367145 + z*(-5089.1530840726 + \
            z*(853.5533353388611 + z*(-133.2587017014444 + (21.0131554401542 - 3.278571068826234*z)*z))) + \
            y*(-270.983805184062 + z*(1552.307223226202 + \
            z*(-589.53765264366 + (115.91861051767 - 10.664504175916349*z)*z)) + \
            y*(1455.0364540468 + z*(-1513.116771538718 + \
            z*(820.438986970584 + z*(-222.2416255268872 + 21.72103359585985*z))) + \
            y*(-672.50778314507 + z*(998.720781638304 + \
            z*(-718.6359919632359 + (195.2050074375488 - 8.31535531044525*z)*z)) + \
            y*(397.968445406972 + z*(-603.630761243752 + (456.589115201523 - 105.4993508931208*z)*z) + \
            y*(-194.618310617595 + y*(63.5113936641785 - 9.63108119393062*y + \
            z*(-44.5794634280918 + 24.511816254543362*z)) + \
            z*(241.04130980405 + z*(-165.8169157020456 + 25.92762672308884*z)))))))        
        
        g08 = x2*(-3310.49154044839 + z*(769.588305957198 + \
            z*(-289.5972960322374 + (63.3632691067296 - 13.1240078295496*z)*z)) + \
            x*(199.459603073901 + x*(-54.7919133532887 + 36.0284195611086*x - 22.6683558512829*y + \
            (-8.16387957824522 - 90.52653359134831*z)*z) + \
            z*(-104.588181856267 + (204.1334828179377 - 13.65007729765128*z)*z) + \
            y*(-175.292041186547 + (166.3847855603638 - 88.449193048287*z)*z + \
            y*(383.058066002476 + y*(-460.319931801257 + 234.565187611355*y) + \
            z*(-108.3834525034224 + 76.9195462169742*z)))) + \
            y*(729.116529735046 + z*(-687.913805923122 + \
            z*(374.063013348744 + z*(-126.627857544292 + 35.23294016577245*z))) + \
            y*(-860.764303783977 + y*(694.244814133268 + \
            y*(-297.728741987187 + (149.452282277512 - 109.46187570047641*z)*z) + \
            z*(-409.779283929806 + (340.685093521782 - 44.5130937305652*z)*z)) + \
            z*(674.819060538734 + z*(-534.943668622914 + (176.8161433232 - 39.600077360584095*z)*z)))))
        # fmt: on

        # Note. This pressure derivative of the gibbs function is in units of (J/kg) (Pa^-1) = m^3/kg
        return (g03 + g08) * 1e-8

    def _dg_dp2(self) -> float:
        """second partial derivative of Gibbs function w.r.t. pressure"""
        # Code based on matlab code from [ZH21]
        x2 = 0.0248826675584615 * self.salinity
        x = np.sqrt(x2)
        y = self.temperature * 0.025
        z = self.pressure * 1e-4

        # fmt: off
        g03 = -5089.1530840726 + z*(1707.1066706777221 + \
            z*(-399.7761051043332 + (84.0526217606168 - 16.39285534413117*z)*z)) + \
            y*(1552.307223226202 + z*(-1179.07530528732 + (347.75583155301 - 42.658016703665396*z)*z) + \
            y*(-1513.116771538718 + z*(1640.877973941168 + z*(-666.7248765806615 + 86.8841343834394*z)) + \
            y*(998.720781638304 + z*(-1437.2719839264719 + (585.6150223126464 - 33.261421241781*z)*z) + \
            y*(-603.630761243752 + (913.178230403046 - 316.49805267936244*z)*z + \
            y*(241.04130980405 + y*(-44.5794634280918 + 49.023632509086724*z) + \
            z*(-331.6338314040912 + 77.78288016926652*z))))))
        
        g08 = x2*(769.588305957198 + z*(-579.1945920644748 + (190.08980732018878 - 52.4960313181984*z)*z) + \
            x*(-104.588181856267 + x*(-8.16387957824522 - 181.05306718269662*z) + \
            (408.2669656358754 - 40.95023189295384*z)*z + \
            y*(166.3847855603638 - 176.898386096574*z + y*(-108.3834525034224 + 153.8390924339484*z))) + \
            y*(-687.913805923122 + z*(748.126026697488 + z*(-379.883572632876 + 140.9317606630898*z)) + \
            y*(674.819060538734 + z*(-1069.887337245828 + (530.4484299696 - 158.40030944233638*z)*z) + \
            y*(-409.779283929806 + y*(149.452282277512 - 218.92375140095282*z) + \
            (681.370187043564 - 133.5392811916956*z)*z))))
        # fmt: on

        # Note. This is the second derivative of the Gibbs function with respect to
        # pressure, measured in Pa.  This derivative has units of (J/kg) (Pa^-2).
        return (g03 + g08) * 1e-16

    def _dg_ds2(self) -> float:
        """second partial derivative of Gibbs function w.r.t. salinity"""
        # Code based on matlab code from [ZH21]
        sfac = 0.0248826675584615
        x2 = sfac * self.salinity
        x = np.sqrt(x2)
        y = self.temperature * 0.025
        z = self.pressure * 1e-4

        # fmt: off
        g08 = 5812.814566267320 + 851.2267349467060*y + x*(-3648.219935726910 \
            + x*(8103.204624147880 + x*(-8187.513078222526 + x*(4495.214854534080 \
            - 850.3093707944657*x))) + y*(-740.1112652125230 + x*(2175.341332000392 \
            + x*(-1470.212300173320 + 441.0859475949660*x)) + y*(-64.59970139670629 \
            - 274.2290036817964*x + y*(-15.03410562928125 + 197.4670779425016*x \
            + y*(1.313400992713418 - 68.55903096791521*x + 9.987880382780312*x*y)))) \
            + z*(299.1894046108515 + x*(-219.1676534131548 + 270.2131467083145*x) \
            + y*(-262.9380617798205 - 90.67342340513160*x + y*(+ 574.5870990037140 \
            + y*(-690.4798977018854 + 351.8477814170325*y))) + z*(-78.44113639220025 \
            - 16.32775915649044*x + y*(124.7885891702729 - 81.28758937756680*y) \
            + z*(102.0667414089689 - 120.7020447884644*x + y*(-44.22459652414350 \
            + 38.45977310848710*y) - 5.118778986619230*z))))
        # fmt: on

        if g08 > 0:
            g08 /= x2
        if g08 == 0:
            g08 = float("NaN")

        return 0.5 * sfac * sfac * g08

    def _dn_dS(self, wavelength: ArrayLike) -> NDArray:
        C = WaterBaseModel._MS_CONSTS
        L = u.convert(np.asarray(wavelength), u.um)
        P = self.pressure
        Tc = self.temperature
        sal_con = 35.16504 / 35  # conversion, SA = sal_con x PSU
        return (
            C.S0
            + C.S1LM2 / L**2
            + C.S1T * Tc
            + C.S1T2 * Tc**2
            + C.S1T3 * Tc**3
            + C.STL * Tc * L
            + P * (C.P1S + C.PTS * Tc + C.PT2S * Tc**2)
        ) / sal_con

    @medium_property
    def scattering_coef(self, wavelength: NDArray) -> NDArray:
        # Code based on matlab code from [ZH21]
        Kbz = 1.3806503e-23  #  Boltzmann constant
        Tk = self.temperature + 273.15  #  Absolute temperature
        lam = u.convert(wavelength, u.m)
        delta = self.depolarizationRatio
        nsw = self.refractive_index(wavelength)
        dnswds = self._dn_dS(wavelength)
        nsw2 = nsw**2
        DFRI = (nsw2 - 1) * (1 + 2 / 3 * (nsw2 + 2) * (nsw / 3 - 1 / 3 / nsw) ** 2)
        SFRI = 2 * nsw * dnswds
        gp = self._dg_dp()
        gpp = self._dg_dp2()
        beta90 = np.pi**2 * Kbz / 2 * (lam ** (-4)) * Tk
        beta90 *= (6 + 6 * delta) / (6 - 7 * delta)  # cabannes factor
        if self.salinity > 0:
            gss = self._dg_ds2()
            beta90 *= DFRI**2 * (-gpp / gp) + SFRI**2 * (gp / gss)
        else:
            beta90 *= DFRI**2.0 * (-gpp / gp)
        beta = 8 * np.pi / 3 * beta90 * (2 + delta) / (1 + delta)
        return beta


class SmithNaturalWaterMeasurements(MediumModel):
    """
    Interpolates Smith's measurements of absorption and scattering coefficients
    for clean natural water. Only valid between 200nm and 800nm.

    References
    ----------
    Raymond C. Smith and Karen S. Baker, "Optical properties of the clearest
    natural waters (200-800 nm)," Appl. Opt. 20, 177-184 (1981)
    """

    # lazily load table
    _TABLE = None

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.restrictWavelengthRange((200.0, 800.0) * u.nm)
        # load table if necessary
        if SmithNaturalWaterMeasurements._TABLE is None:
            SmithNaturalWaterMeasurements._TABLE = loadCSV("water_smith81.csv")

    @medium_property
    def absorption_coef(self, wavelength: NDArray) -> NDArray:
        table = SmithNaturalWaterMeasurements._TABLE
        assert table is not None
        return np.interp(wavelength, table[:, 0], table[:, 1])

    @medium_property
    def scattering_coef(self, wavelength: NDArray) -> NDArray:
        table = SmithNaturalWaterMeasurements._TABLE
        assert table is not None
        return np.interp(wavelength, table[:, 0], table[:, 2])


class KokhanovskyOceanWaterPhaseMatrix(MediumModel):
    """
    Empirical model for the phase matrix of ocean water as described by
    Kokhanovsky[1].

    Parameters
    ----------
    p90: float
        Degree of polarization at 90°
    theta0: float
        Shift in angle
    alpha: float
        multipole scatter slope
    xi: float
        multipole scatter scale

    References
    ----------
    [1] Alexander A. Kokhanovsky. "Parameterization of the Mueller matrix of oceanic
        waters". Journal of Geophysical Research, Vol. 108, No. C6, 3175,
        doi:10.1029/2001JC001222, 2003
    """

    def __init__(
        self, p90: float, theta0: float, alpha: float, xi: float, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.p90 = p90
        self.theta0 = theta0
        self.alpha = alpha
        self.xi = xi

    @medium_property(range=(-1.0, 1.0))
    def phase_m12(self, cos_theta: ArrayLike) -> NDArray:
        """m12 element of the phase matrix"""
        cos_theta_sq = np.square(cos_theta)
        sin_theta_sq = 1.0 - cos_theta_sq
        return -self.p90 * sin_theta_sq / (1.0 + self.p90 * cos_theta_sq)

    @medium_property(range=(-1.0, 1.0))
    def phase_m22(self, cos_theta: ArrayLike) -> NDArray:
        """m22 element of the phase matrix"""
        theta = np.arccos(cos_theta)
        z = theta - self.theta0
        cos_z_sq = np.square(np.cos(z))
        e = self.xi * np.exp(-self.alpha * theta)
        return (self.p90 * (1.0 + cos_z_sq) + e) / (1.0 + self.p90 * cos_z_sq + e)

    @medium_property(range=(-1.0, 1.0))
    def phase_m33(self, cos_theta: ArrayLike) -> NDArray:
        """m33 element of the phase matrix"""
        cos_theta = np.asarray(cos_theta)
        theta = np.arccos(cos_theta)
        ct_sq = np.square(cos_theta)
        e = self.xi * np.exp(-self.alpha * theta)
        return (2 * self.p90 * cos_theta + e) / (1.0 + self.p90 * ct_sq + e)
