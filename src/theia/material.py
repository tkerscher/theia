from __future__ import annotations

import re
import warnings

import hephaistos as hp
import numpy as np

import scipy.constants
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.stats.sampling import NumericalInversePolynomial

from dataclasses import dataclass, field
from enum import Enum, IntFlag
from itertools import chain
from pathlib import Path
from struct import pack, unpack
from types import SimpleNamespace
from zipfile import ZipFile, Path as ZipPath

from theia.compiler import createPreamble
from theia.lookup import createTableFromFunction, Interpolation
from theia.property import (
    FloatProperty,
    Property,
    PropertyTable,
    PropertyTableEntry,
    TableProperty,
    createSlotMacros,
    loadTableEntry,
    saveTableEntry,
)
from theia.surface import SurfaceModel
from theia.volume import Transparent, VolumeModel
from theia.util import classproperty, intersectRange, loadCSV
import theia.units as u

from collections.abc import Callable, Iterable, Iterator, Mapping
from numpy.typing import ArrayLike, NDArray
from typing import Any, Final, Generic, TypeVar, overload
from typing_extensions import Self


__all__ = [
    "getPropertySamples",
    "loadMaterials",
    "saveMaterials",
    "speed_of_light",
    "DefaultMaterialSlots",
    "DefaultMediaSlots",
    "Material",
    "MaterialFlags",
    "MaterialStore",
    "Medium",
    "OpticalProperties",
    "parseMaterialFlags",
    "VACUUM_IDX",
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
    physicModel: VolumeModel = field(default_factory=Transparent)
    """Physical model describing the physical interaction of rays within this medium"""


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
            Nn, Ng = np.size(n), np.size(vg)
            # constant?
            if Nn != 1:
                # c / vg = n - lambda * (d_n/d_lambda)
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

    TRANSMIT_HIT = 0x200
    """
    Indicates to the surface model, that it should use the transmitted ray to
    create hits. If instead this bit is not set, the surface model may use the
    incident or absorbed ray. The exact behavior depends on the specific surface
    model.
    """


_materialFlagsMap = {
    "B": MaterialFlags.BLACK_BODY,
    "D": MaterialFlags.DETECTOR,
    "Dt": MaterialFlags.DETECTOR | MaterialFlags.TRANSMIT_HIT,
    "L": MaterialFlags.LIGHT_SOURCE,
    "Lt": MaterialFlags.LIGHT_SOURCE | MaterialFlags.TRANSMIT_HIT,
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
    respectively. Similarly, `D` and `L` can be appended with `t` to set the
    `TRANSMIT_HIT` bit.

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
        physicModel: SurfaceModel,
        *,
        flags: (
            tuple[MaterialFlags | str, MaterialFlags | str] | MaterialFlags | str
        ) = MaterialFlags(0),
        properties: Mapping[str, Property] | None = None,
    ) -> None:
        self.name = name
        self.inside = inside
        self.outside = outside
        self.physicModel = physicModel
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
    physicModel: SurfaceModel
    """
    Model describing the physical interaction of rays as they interact with a
    surface of this material.
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
    for name, med in media_dict.items():
        med_path = file.joinpath(f"media/{name}")
        # write properties
        prop_path = med_path.joinpath("properties/")
        saveTableEntry(prop_path, _createTableEntryFromMedium(med))
        # write volume model
        med.physicModel.save(med_path.joinpath(f"physicmodel/{med.physicModel.name}"))

    # save all material
    getName = lambda m: m.name if isinstance(m, Medium) else m
    printName = lambda m: m if (m := getName(m)) is not None else ""
    for mat in material:
        mat_path = file.joinpath(f"material/{mat.name}/")
        # write material description
        with mat_path.joinpath("material").open("w") as mat_file:
            mat_file.write(f"{str(int(mat.flagsInward))},{printName(mat.inside)}\n")
            mat_file.write(f"{str(int(mat.flagsOutward))},{printName(mat.outside)}\n")
        # write properties
        saveTableEntry(mat_path.joinpath("properties/"), mat.properties)
        # write surface model
        mat.physicModel.save(mat_path.joinpath(f"physicmodel/{mat.physicModel.name}"))


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

    def getPhysicModelFile(dir: ZipPath) -> ZipPath:
        content = list(dir.iterdir())
        if len(content) != 1 or not content[0].is_file():
            raise ValueError(
                f'Unsupported file structure in "{dir}". Expected single file'
            )
        return content[0]

    # load media
    med_dict = {}
    for med_path in path.joinpath("media/").iterdir():
        if not med_path.is_dir():
            raise ValueError(f'Unexpected file "{med_path}"')
        name = med_path.name
        # load properties
        props = dict(loadTableEntry(med_path.joinpath("properties/")))
        if "wavelength_range" not in props:
            raise ValueError(
                f'Media "{name}" is missing required property "wavelength_range"'
            )
        wavelengthRange = props.pop("wavelength_range")
        if not isinstance(wavelengthRange, FloatProperty):
            raise ValueError(
                f'Property "wavelength_range" of media "{name}" has wrong type!'
            )
        wavelengthRange = wavelengthRange.value
        # load volume model
        physic_file = getPhysicModelFile(med_path.joinpath("physicmodel/"))
        physic = VolumeModel.from_file(physic_file)
        # create and save loaded medium
        med_dict[name] = Medium(name, wavelengthRange, props, physic)

    def parseLine(line: str):
        flags, med = line.split(",", 1)
        flags = MaterialFlags(int(flags))
        med = med[:-1] or None  # remove new line and replace empty with None
        if med in med_dict:
            med = med_dict[med]
        return (flags, med)

    # load materials
    mat_dict = {}
    for mat_path in path.joinpath("material/").iterdir():
        if not mat_path.is_dir():
            raise ValueError(f'Unexpected file "{mat_path}"')
        name = mat_path.name
        # load material description file
        desc_file = mat_path.joinpath("material")
        if not desc_file.exists() or not desc_file.is_file():
            raise ValueError(f'Material "{name}" misses description file')
        with desc_file.open("r") as file:
            flagsIn, medIn = parseLine(file.readline())
            flagsOut, medOut = parseLine(file.readline())
        flags = (flagsIn, flagsOut)
        # load properties
        props = loadTableEntry(mat_path.joinpath("properties/"))
        # load surface model
        physic_file = getPhysicModelFile(mat_path.joinpath("physicmodel/"))
        physic = SurfaceModel.from_file(physic_file)
        # create and save loaded material
        mat_dict[name] = Material(
            name, medIn, medOut, physic, flags=flags, properties=props
        )

    # done
    return mat_dict, med_dict


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
        # add mandatory slots
        mediaSlots = set() if mediaSlots is None else set(mediaSlots)
        mediaSlots.add("wavelength_range")
        materialSlots = set() if materialSlots is None else set(materialSlots)
        materialSlots.update(["inwards", "outwards"])
        # collect all media
        mediaDict = {m.name: m for m in media}
        for med in chain.from_iterable((m.inside, m.outside) for m in material):
            if isinstance(med, Medium) and med.name not in mediaDict:
                mediaDict[med.name] = med
        # collect all volume models
        # in order to handle vacuum more easily in the tracer,
        # ensure the transparent model is always present at index 0
        volModelSet = {m.physicModel for m in mediaDict.values()}
        volModelSet.discard(Transparent())  # add later at index 0
        self._volumeModels = [Transparent(), *volModelSet]
        idxMap = {model: i for i, model in enumerate(self._volumeModels)}
        self._volumeModelIndices = [idxMap[m.physicModel] for m in mediaDict.values()]
        # create media table
        mediaProps = {n: _createTableEntryFromMedium(m) for n, m in mediaDict.items()}
        self._mediaTable = PropertyTable(mediaProps, requiredSlots=mediaSlots)
        # create mapping from medium index to volume model index
        if len(mediaDict) > 0:
            volMap = np.ascontiguousarray(self._volumeModelIndices, dtype=np.uint32)
            self._volumeMap = hp.Tensor(volMap)
        else:
            # we cannot create an empty tensor
            self._volumeMap = None

        # collect surface models
        material = list(material)
        self._surfaceModels = list({m.physicModel for m in material})
        idxMap = {model: i for i, model in enumerate(self._surfaceModels)}
        self._surfaceModelIndices = [idxMap[m.physicModel] for m in material]
        # create material table
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
    def surfaceModels(self) -> list[SurfaceModel]:
        """Ordered list of surface models referenced by the materials"""
        return self._surfaceModels

    @property
    def surfaceModelMap(self) -> list[int]:
        """Mapping from material index to surface model index"""
        return self._surfaceModelIndices

    @property
    def volumeModels(self) -> list[VolumeModel]:
        """Ordered list of volume models referenced by the media"""
        return self._volumeModels

    @property
    def volumeModelMap(self) -> list[int]:
        """Mapping from medium index to volume model index"""
        return self._volumeModelIndices

    @property
    def volumeModelMapTensor(self) -> hp.Tensor | None:
        """Tensor containing the volume module map on the device"""
        return self._volumeMap

    @property
    def slotMacros(self) -> dict[str, int]:
        """Dictionary containing the slot names to be used in shaders"""
        mediaSlots = createSlotMacros(self.media, "MEDIA_SLOT")
        materialSlots = createSlotMacros(self.materials, "MATERIAL_SLOT")
        return mediaSlots | materialSlots

    def getSurfaceModel(self, material: str | int) -> SurfaceModel:
        """Returns the surface model referenced by the given surface"""
        if isinstance(material, str):
            material = self.materials[material]
        idx = self.surfaceModelMap[material]
        return self.surfaceModels[idx]

    def getVolumeModel(self, medium: str | int) -> VolumeModel:
        """Returns the volume model referenced by the given volume"""
        if isinstance(medium, str):
            medium = self.media[medium]
        idx = self.volumeModelMap[medium]
        return self.volumeModels[idx]

    def bindParams(self, program: hp.Program | hp.RayTracingPipeline) -> None:
        program.bindParams(
            MediaTable=self.media.tensor,
            MaterialTable=self.materials.tensor,
            MediumMap=self.volumeModelMapTensor,
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
