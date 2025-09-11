from __future__ import annotations

import warnings

import hephaistos as hp
import numpy as np
import scipy.constants
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.stats.sampling import NumericalInversePolynomial

from ctypes import Structure, c_float, c_uint32, c_uint64, sizeof
from types import MappingProxyType, SimpleNamespace

from theia.lookup import Table, getTableSize
from theia.util import loadCSV, intersectRange
import theia.units as u

import json
import re
from jsonschema import validate
from pathlib import Path
from zipfile import Path as ZipPath, ZipFile, is_zipfile

from collections.abc import Iterable
from enum import IntFlag
from io import TextIOBase
from os import PathLike
from numpy.typing import ArrayLike, NDArray
from typing import Final


__all__ = [
    "checkMedium",
    "loadMaterials",
    "parseMaterialFlags",
    "saveMaterials",
    "speed_of_light",
    "BK7Model",
    "FournierForandPhaseFunction",
    "HenyeyGreensteinPhaseFunction",
    "KokhanovskyOceanWaterPhaseMatrix",
    "Material",
    "MaterialFlags",
    "MaterialStore",
    "Medium",
    "MediumModel",
    "NumericalPhaseSamplingMixin",
    "PureWaterModel",
    "RayleighScatteringModel",
    "RayleighScatteringPhaseFunction",
    "SellmeierEquation",
    "SmithNaturalWaterMeasurements",
    "WaterBaseModel",
]


def __dir__():
    return __all__


#################################### COMMON ####################################


speed_of_light: Final[float] = 1.0 * u.c
"""speed of light in internal units"""


class Medium:
    """
    Class to describe a named medium by storing the physical properties as
    tables each containing values of its respective property sampled at
    equidistant positions on a specified range of wavelengths.

    Parameters
    ----------
    name: str
        Unique name of the medium. Can later be used to fetch the address of
        the medium in gpu local memory.
    lambda_min: float
        Lower limit of the range of wavelengths the tables are defined
    lambda_max: float
        Upper limit of the range of wavelengths the tables are defined
    refractive_index: ArrayLike | None, default = None
        Table of refractive index as function of wavelength.
        None defaults to a constant value of 1.0.
    group_velocity: ArrayLike | None, default = None
        Table of group velocity as function of wavelength.
        None defaults to a constant value of `c`.
    absorption_coef: ArrayLike | None, default = None
        Table of absorption coefficient as function of wavelength.
        None defaults to a constant value of 0.0.
    scattering_coef: ArrayLike | None, default = None
        Table of scattering coefficient as function of wavelength.
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
    phase_m12: ArrayLike | None, default = None
        Table of normalized m12 element of the phase matrix as function of cos
        theta. Constant zero if None.
    phase_m22: ArrayLike | None, default = None
        Table of normalized m22 element of the phase matrix as function of cos
        theta. Constant zero if None.
    phase_m33: ArrayLike | None, default = None
        Table of normalized m33 element of the phase matrix as function of cos
        theta. Constant zero if None.
    phase_m34: ArrayLike | None, default = None
        Table of normalized m34 element of the phase matrix as function of cos
        theta. Constant zero if None.

    Note
    ----
    The phase matrix and its elements m12, m33, m34 govern how polarization as
    described by the Stokes' vector changes after volumetric scattering. We
    assume some symmetry in the medium causing the matrix to be of the following
    form (note it's normalized to the volume scattering function)
        /  1  m12  0   0  \\  \n
        | m12 m22  0   0  |   \n
        |  0   0  m33 m34 |   \n
       \\  0   0 -m34 m33 /   \n
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
            ("phase_m12", c_uint64),  # Table1D
            ("phase_m22", c_uint64),  # Table1D
            ("phase_m33", c_uint64),  # Table1D
            ("phase_m34", c_uint64),  # Table1D
        ]

    ALIGNMENT: Final[int] = 8
    """Alignment of the GLSL structure in device memory"""

    def __init__(
        self,
        name: str,
        lambda_min: float,
        lambda_max: float,
        *,
        refractive_index: ArrayLike | None = None,
        group_velocity: ArrayLike | None = None,
        absorption_coef: ArrayLike | None = None,
        scattering_coef: ArrayLike | None = None,
        log_phase_function: ArrayLike | None = None,
        phase_sampling: ArrayLike | None = None,
        phase_m12: ArrayLike | None = None,
        phase_m22: ArrayLike | None = None,
        phase_m33: ArrayLike | None = None,
        phase_m34: ArrayLike | None = None,
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
        self.phase_m12 = phase_m12
        self.phase_m22 = phase_m22
        self.phase_m33 = phase_m33
        self.phase_m34 = phase_m34

    @property
    def name(self) -> str:
        """
        Name of this medium. Can later be used to fetch the address of the
        medium in gpu local memory.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def lambda_min(self) -> float:
        """Minimum wavelength for which the optical properties are defined"""
        return self._lambda_min

    @lambda_min.setter
    def lambda_min(self, value: float) -> None:
        self._lambda_min = value

    @property
    def lambda_max(self) -> float:
        """Maximum wavelength for which the optical properties are defined"""
        return self._lambda_max

    @lambda_max.setter
    def lambda_max(self, value: float) -> None:
        self._lambda_max = value

    @property
    def refractive_index(self) -> ArrayLike | None:
        """
        Table containing values of the refractive as a function of wavelength
        sampled at equidistant points on the range defined by lambda min/max.
        If None, a constant value of 1.0 is assumed.
        """
        return self._refractive_index

    @refractive_index.setter
    def refractive_index(self, value: ArrayLike | None) -> None:
        self._refractive_index = value

    @refractive_index.deleter
    def refractive_index(self) -> None:
        self._refractive_index = None

    @property
    def group_velocity(self) -> ArrayLike | None:
        """
        Table containing values of the group velocity as a function of
        wavelength sampled at equidistant points on the range defined by lambda
        min/max.
        If None, a constant value of `c` is assumed.
        """
        return self._group_velocity

    @group_velocity.setter
    def group_velocity(self, value: ArrayLike | None) -> None:
        self._group_velocity = value

    @group_velocity.deleter
    def group_velocity(self) -> None:
        self._group_velocity = None

    @property
    def absorption_coef(self) -> ArrayLike | None:
        """
        Table containing values of the absorption coefficient as a function of
        wavelength sampled at equidistant points on the range defined by lambda
        min/max.
        If None, a constant value of 0.0 is assumed.
        """
        return self._absorption_coef

    @absorption_coef.setter
    def absorption_coef(self, value: ArrayLike | None) -> None:
        self._absorption_coef = value

    @absorption_coef.deleter
    def absorption_coef(self) -> None:
        self._absorption_coef = None

    @property
    def scattering_coef(self) -> ArrayLike | None:
        """
        Table containing values of the scattering coefficient as a function of
        wavelength sampled at equidistant points on the range defined by lambda
        min/max.
        If None, a constant value of 0.0 is assumed.
        """
        return self._scattering_coef

    @scattering_coef.setter
    def scattering_coef(self, value: ArrayLike | None) -> None:
        self._scattering_coef = value

    @scattering_coef.deleter
    def scattering_coef(self) -> None:
        self._scattering_coef = None

    @property
    def log_phase_function(self) -> ArrayLike | None:
        """
        Table of logarithmic scattering phase function as a function of the
        cosine of the angle between incoming and outgoing ray in radians over
        the range [0,1].
        None defaults to a constant value of 1, i.e. uniform scattering.
        """
        return self._phase_function

    @log_phase_function.setter
    def log_phase_function(self, value: ArrayLike | None) -> None:
        self._phase_function = value

    @log_phase_function.deleter
    def log_phase_function(self) -> None:
        self._phase_function = None

    @property
    def phase_sampling(self) -> ArrayLike | None:
        """
        Table containing values of the inverse cumulative density function of
        the phase function used for importance sampling.
        If None, sampling happens uniform random.
        """
        return self._phase_sampling

    @phase_sampling.setter
    def phase_sampling(self, value: ArrayLike | None) -> None:
        self._phase_sampling = value

    @phase_sampling.deleter
    def phase_sampling(self) -> None:
        self._phase_sampling = None

    @property
    def phase_m12(self) -> ArrayLike | None:
        """
        Table of normalized m12 element of the phase matrix as function of cos
        theta. Constant zero if None.
        """
        return self._phase_m12

    @phase_m12.setter
    def phase_m12(self, value: ArrayLike | None) -> None:
        self._phase_m12 = value

    @phase_m12.deleter
    def phase_m12(self) -> None:
        self._phase_m12 = None

    @property
    def phase_m22(self) -> ArrayLike | None:
        """
        Table of normalized m22 element of the phase matrix as function of cos
        theta. Constant zero if None.
        """
        return self._phase_m22

    @phase_m22.setter
    def phase_m22(self, value: ArrayLike | None) -> None:
        self._phase_m22 = value

    @phase_m22.deleter
    def phase_m22(self) -> None:
        self._phase_m22 = None

    @property
    def phase_m33(self) -> ArrayLike | None:
        """
        Table of normalized m33 element of the phase matrix as function of cos
        theta. Constant zero if None.
        """
        return self._phase_m33

    @phase_m33.setter
    def phase_m33(self, value: ArrayLike | None) -> None:
        self._phase_m33 = value

    @phase_m33.deleter
    def phase_m33(self) -> None:
        self._phase_m33 = None

    @property
    def phase_m34(self) -> ArrayLike | None:
        """
        Table of normalized m34 element of the phase matrix as function of cos
        theta. Constant zero if None.
        """
        return self._phase_m34

    @phase_m34.setter
    def phase_m34(self, value: ArrayLike | None) -> None:
        self._phase_m34 = value

    @phase_m34.deleter
    def phase_m34(self) -> None:
        self._phase_m34 = None

    # list containing names of arrays/properties used by serialization
    _PROPS = [
        "refractive_index",
        "group_velocity",
        "absorption_coef",
        "scattering_coef",
        "log_phase_function",
        "phase_sampling",
        "phase_m12",
        "phase_m22",
        "phase_m33",
        "phase_m34",
    ]

    def save(self, file) -> None:
        """
        Serializes the medium and saves it at the given location.
        Produces a file in numpy's `.npz` format.

        Parameters
        ----------
        file: str or file
            Path or open file where the data will be saved.
        """
        # decline files opened in text mode
        if isinstance(file, TextIOBase):
            raise ValueError("file must be opened in binary mode!")
        # collect all non empty arrays
        arrays = {
            p: getattr(self, p) for p in self._PROPS if getattr(self, p) is not None
        }
        # add lambda range
        arrays["lambda_range"] = np.array([self.lambda_min, self.lambda_max])

        np.savez(file, **arrays)

    @staticmethod
    def load(file, *, name: str = "unnamed") -> Medium:
        """
        Loads the serialized medium from the given file or path.

        Parameters
        ----------
        file: str or file
            Path or open file where the data is stored.
        name: str, default="unnamed"
            Name of the loaded medium.

        Returns
        -------
        Deserialized medium loaded from the given file or path.
        """
        # decline files opened in text mode
        if isinstance(file, TextIOBase):
            raise ValueError("file must be opened in binary mode!")
        data = np.load(file)
        lam = data.get("lambda_range")
        if lam is None or lam.shape != (2,):
            raise ValueError("File does not contain valid lambda range!")

        medium = Medium(name, lam[0], lam[1])
        for prop in Medium._PROPS:
            setattr(medium, prop, data.get(prop))

        return medium


def checkMedium(medium: Medium, *, raises: bool = True, rtol=5e-3, atol=1e-5) -> bool:
    """
    Runs a series of basic tests on the given medium for physical plausibility
    and returns `True` if it is successfull. On failure, if `raises` is `True`
    raises an `AssertionError` or returns `False` otherwise.

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
    # bit ugly, but nicer than duplicating the same if/else over and over
    try:
        # fmt: off
        assert medium.lambda_max >= medium.lambda_min
        if medium.refractive_index is not None:
            assert medium.group_velocity is not None, "Refractive index was specified but no group velocity!"
            assert np.ndim(medium.refractive_index) == 1, "Refractive index must be 1D!"
            assert np.min(medium.refractive_index) >= 1.0, "Refractive index must be at least 1.0!"
        if medium.group_velocity is not None:
            assert medium.refractive_index is not None, "Group velocity was specified, but no refractive index!"
            assert np.ndim(medium.group_velocity) == 1, "Group velocity must be 1D!"
            # c / vg = n - lambda * (d_n/d_lambda)
            Nn = np.size(medium.refractive_index)
            n = medium.refractive_index
            lam_n = np.linspace(medium.lambda_min, medium.lambda_max, Nn)
            dl = (medium.lambda_max - medium.lambda_min) / (Nn - 1)
            dn_dl = np.gradient(n, dl)
            vg = speed_of_light / (n - lam_n * dn_dl)
            Ng = np.size(medium.group_velocity)
            lam_g = np.linspace(medium.lambda_min, medium.lambda_max, Ng)
            vg = np.interp(lam_g, lam_n, vg)
            assert np.allclose(vg, medium.group_velocity, rtol=rtol, atol=atol), "Group velocity does not match refractive index!"
        if medium.absorption_coef is not None:
            assert np.ndim(medium.absorption_coef) == 1, "Absorption coefficient must be 1D!"
            assert np.min(medium.absorption_coef) >= 0.0, "Absorption coefficient must not be negative!"
        if medium.scattering_coef is not None:
            assert np.ndim(medium.scattering_coef) == 1, "Scattering coefficient must be 1D!"
            assert np.min(medium.scattering_coef) >= 0.0, "Scattering coefficient must not be negative!"
        if medium.phase_sampling is not None:
            assert np.ndim(medium.phase_sampling) == 1, "Phase sampling function must be 1D!"
            assert medium.log_phase_function is not None, "Phase sampling is provided, but no phase function!"
            f = medium.phase_sampling
            assert np.all((f >= -1) & (f <= 1)), "Domain of phase sampling must be in [-1, 1]!"
        if medium.log_phase_function is not None:
            assert np.ndim(medium.log_phase_function) == 1, "Phase function must be 1D!"
            assert medium.phase_sampling is not None, "Phase function is defined, but no sampling!"
            N = np.size(medium.log_phase_function)
            y = np.exp(medium.log_phase_function)
            dx = 2.0 / (N - 1)
            cdf = 2 * np.pi * cumulative_simpson(y, dx=dx, initial=0)
            assert np.allclose(cdf[-1], 1.0, rtol=rtol, atol=atol), "Phase function must integrate over solid angles to 1!"
            u = np.linspace(-1, 1, N)
            quantiles = np.interp(medium.phase_sampling, u, cdf)
            # sampling is fine in both directions, i.e. both u and 1-u are fine
            # this will only flip the sign in the difference -> use abs
            # if the sampling table is correct, the quantiles should be equidistant
            dq = np.abs(np.diff(quantiles))
            du = 1 / len(dq)
            assert np.allclose(dq, du, rtol=rtol, atol=atol), "Phase sampling function does not reproduce phase function!"
        if medium.phase_m12 is not None:
            assert np.ndim(medium.phase_m12) == 1, "Phase function m12 must be 1D!"
            m = medium.phase_m12
            assert np.all((m >= -1.0) & (m <= 1.0)), "Mueller matrix must be normalized!"
        if medium.phase_m22 is not None:
            assert np.ndim(medium.phase_m22) == 1, "Phase function m22 must be 1D!"
            m = medium.phase_m22
            assert np.all((m >= -1.0) & (m <= 1.0)), "Mueller matrix must be normalized!"
        if medium.phase_m33 is not None:
            assert np.ndim(medium.phase_m33) == 1, "Phase function m33 must be 1D!"
            m = medium.phase_m33
            assert np.all((m >= -1.0) & (m <= 1.0)), "Mueller matrix must be normalized!"
        if medium.phase_m34 is not None:
            assert np.ndim(medium.phase_m34) == 1, "Phase function m34 must be 1D!"
            m = medium.phase_m34
            assert np.all((m >= -1.0) & (m <= 1.0)), "Mueller matrix must be normalized!"
        # fmt: on
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
    """

    # tokenize
    tokens = re.findall(r"[A-Z][a-z]*", flags)
    # parse tokens
    result = MaterialFlags.NO_REFLECT | MaterialFlags.NO_TRANSMIT
    for token in tokens:
        if token in _materialFlagsMap:
            result ^= _materialFlagsMap[token]
        else:
            raise ValueError(f"Unknown material flag '{token}'")
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
        inside: Medium | str | None,
        outside: Medium | str | None,
        *,
        flags: (
            tuple[MaterialFlags | str, MaterialFlags | str] | MaterialFlags | str
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
    def inside(self) -> Medium | str | None:
        """
        Medium in the inside of a geometry. Can also be specified by its name
        which will get resolved during baking/serialization.
        None defaults to vacuum.
        """
        return self._inside

    @inside.setter
    def inside(self, value: Medium | str | None) -> None:
        self._inside = value

    @property
    def outside(self) -> Medium | str | None:
        """
        Medium in the outside of a geometry. Can also be specified by its name
        which will get resolved during baking/serialization.
        None defaults to vacuum.
        """
        return self._outside

    @outside.setter
    def outside(self, value: Medium | str | None) -> None:
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


# fmt: off
_materialJsonSchema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "inside": {"type": ["string", "null"]},
            "outside": {"type": ["string", "null"]},
            "flagsInward": {"type": "number", "minimum": 0},
            "flagsOutward": {"type": "number", "minimum": 0},            
        },
        "required": ["name", "inside", "outside", "flagsInward", "flagsOutward"],
        "additionalProperties": False,
    },
}
# fmt: on


def saveMaterials(
    path: str | bytes | PathLike,
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
    # convert to path to make things easier
    if not isinstance(path, (Path, ZipPath)):
        return saveMaterials(Path(path), material, media=media)
    # create a zip archive if path does not exist
    if not path.is_dir() and not path.exists():
        with ZipFile(path, "w") as zip:
            return saveMaterials(ZipPath(zip), material, media=media)
    # at this point we expect path to point to a directory (either real or zip)
    if not path.is_dir():
        raise ValueError("Invalid path!")

    # collect all media
    mediaDict = {m.name: m for m in media}
    # first pass: collect all referenced media in material
    for mat in material:
        if isinstance(mat.inside, Medium) and mat.inside.name not in mediaDict:
            mediaDict[mat.inside.name] = mat.inside
        if isinstance(mat.outside, Medium) and mat.outside.name not in mediaDict:
            mediaDict[mat.outside.name] = mat.outside
    # second pass: check if all name references are met
    for mat in material:
        if type(mat.inside) is str and mat.inside not in mediaDict:
            raise ValueError(
                f"Material '{mat.name}' references an unknown medium '{mat.inside}'!"
            )
        if type(mat.outside) is str and mat.outside not in mediaDict:
            raise ValueError(
                f"Material '{mat.name}' references an unknown medium '{mat.outside}'!"
            )

    # first save all media
    mediaDir = path.joinpath("media/")
    if not isinstance(mediaDir, ZipPath):
        mediaDir.mkdir(exist_ok=True)
    for name, medium in mediaDict.items():
        with mediaDir.joinpath(name).open("wb") as file:
            medium.save(file)

    # build representation of materials suitable for serialization
    def serializeMedium(med: Medium | str | None):
        if isinstance(med, Medium):
            return med.name
        else:
            return med

    def serializeMaterial(mat: Material):
        return {
            "name": mat.name,
            "inside": serializeMedium(mat.inside),
            "outside": serializeMedium(mat.outside),
            "flagsInward": int(mat.flagsInward),
            "flagsOutward": int(mat.flagsOutward),
        }

    matList = [serializeMaterial(mat) for mat in material]
    # save material descriptions
    matFile = path.joinpath("material.json")
    with matFile.open("w") as file:
        json.dump(matList, file)


def loadMaterials(
    path: str | bytes | PathLike, *, skipValidation: bool = False
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
    # convert to path to make things easier
    if not isinstance(path, (Path, ZipPath)):
        return loadMaterials(Path(path))
    # check if path points to a zip file
    if path.is_file():
        if is_zipfile(path):
            # open zip file and try again
            with ZipFile(path) as arc:
                return loadMaterials(ZipPath(arc))
        else:
            raise ValueError("Unexpected file format!")
    # at this point we expect path to point to a directory (or zip root)
    if not path.is_dir():
        raise ValueError("Incompatible path!")

    # begin with loading all media stored in the subdirectory "media"
    mediaDir = path.joinpath("media")
    if not mediaDir.exists() or not mediaDir.is_dir():
        raise ValueError('Subdirectory "media" is missing!')
    mediaDict = {}
    for file in mediaDir.iterdir():
        # encapsulate loading media in try/catch for better error message
        try:
            with file.open("rb") as f:
                mediaDict[file.stem] = Medium.load(f, name=file.stem)
        except Exception as ex:
            # needs to a bit funky to handle zip paths correct
            absPath = Path.cwd() / str(file)
            raise ValueError(f"Failed to load media file: {absPath}")

    # next load the material description
    matFile = path.joinpath("material.json")
    if not matFile.is_file():
        raise ValueError('Missing file: "material.json"!')
    # try to load material file
    materialJson = None
    with matFile.open() as file:
        try:
            materialJson = json.load(file)
        except Exception as ex:
            raise ValueError('Invalid material description file: "material.json"!')
    if not skipValidation:
        validate(materialJson, _materialJsonSchema)
    # parse material descriptions
    matDict = {}

    def getMedium(mat: str, med: str | None) -> Medium | None:
        if med is None:
            return None
        if med not in mediaDict:
            raise ValueError(f"Material '{mat}' references unknown medium: '{med}'!")
        else:
            return mediaDict[med]

    for mat in materialJson:
        name = mat["name"]
        if name in matDict:
            raise ValueError(f"Duplicate material '{name}'!")
        inside = getMedium(name, mat["inside"])
        outside = getMedium(name, mat["outside"])
        flagsInward = MaterialFlags(mat["flagsInward"])
        flagsOutward = MaterialFlags(mat["flagsOutward"])
        flags = (flagsInward, flagsOutward)
        matDict[name] = Material(name, inside, outside, flags=flags)

    # Done
    return matDict, mediaDict


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
        self._table_ptr: dict[str, tuple[int, int]] = {}  # (ptr, size)
        self._media_ptr: dict[str, int] = {}
        self._mat_ptr: dict[str, int] = {}

        def allocTable(name: str, data):
            # assume we only add new tables
            if data is not None:
                size = getTableSize(data)
                self._table_ptr[name] = (alloc(size, Table.ALIGNMENT), size)

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
            allocTable(f"{name}_m12", medium.phase_m12)
            allocTable(f"{name}_m22", medium.phase_m22)
            allocTable(f"{name}_m33", medium.phase_m33)
            allocTable(f"{name}_m34", medium.phase_m34)

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
        self._tensor = hp.Tensor(size, mapped=(not freeze))
        self._buffer = hp.Buffer(size)
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
            if isinstance(medium, Medium) and medium.name not in processed_media:
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
        table = Table(data)
        table.copy(ptr)
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
        name = medium.name
        glsl.n = self._updateTable(f"{name}_n", medium.refractive_index)
        glsl.vg = self._updateTable(f"{name}_vg", medium.group_velocity)
        glsl.mu_a = self._updateTable(f"{name}_mua", medium.absorption_coef)
        glsl.mu_s = self._updateTable(f"{name}_mus", medium.scattering_coef)
        glsl.log_phase = self._updateTable(f"{name}_lpf", medium.log_phase_function)
        glsl.phase_sampling = self._updateTable(f"{name}_ps", medium.phase_sampling)
        glsl.phase_m12 = self._updateTable(f"{name}_m12", medium.phase_m12)
        glsl.phase_m22 = self._updateTable(f"{name}_m22", medium.phase_m22)
        glsl.phase_m33 = self._updateTable(f"{name}_m33", medium.phase_m33)
        glsl.phase_m34 = self._updateTable(f"{name}_m34", medium.phase_m34)

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
            Wether to also update referenced media. Media referenced by name
            will be ignored.
        """
        if self.frozen:
            raise RuntimeError("Cannot update frozen MaterialStore")
        if material.name not in self._mat_ptr:
            raise ValueError(
                f"Material {material.name} has not been previously allocated"
            )

        # fetch names of references media
        inside, outside = material.inside, material.outside
        if isinstance(inside, Medium):
            inside = inside.name
        if isinstance(outside, Medium):
            outside = outside.name

        if inside is not None and inside not in self.media:
            raise ValueError(
                f"Material {material.name} references unknown medium {inside}"
            )
        if outside is not None and outside not in self.media:
            raise ValueError(
                f"Material {material.name} references unknown medium {outside}"
            )
        # fetch header
        glsl = Material.GLSL.from_address(self._mat_ptr[material.name])
        glsl.inside = 0 if inside is None else self.media[inside]
        glsl.outside = 0 if outside is None else self.media[outside]
        glsl.flagsInwards = material.flagsInward
        glsl.flagsOutwards = material.flagsOutward

        # update medium if specified
        if updateMedia:
            if isinstance(material.inside, Medium):
                self.updateMedium(material.inside)
            if isinstance(material.outside, Medium):
                self.updateMedium(material.outside)


#################################### MODELS ####################################


class MediumModel:
    """
    Base class for models of media. Implements a function for creating a medium
    by sampling functions provided by base classes.

    Parameters
    ----------
    name: str, default="noname"
        Name of the model. Will be used as default name for instanciated media.
    wavelengthRange: (float, float), default=(0,inf)
        Range of wavelengths in nm for which the model is valid.
    """

    def __init__(
        self,
        *args,
        name: str = "noname",
        wavelengthRange: tuple[float, float] = (100.0, 1500.0),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.wavelengthRange = wavelengthRange

    @property
    def name(self) -> str:
        """Name of the model"""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

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

    def checkWavelength(self, wavelength: ArrayLike) -> NDArray:
        """
        Checks whether the given wavelengths fall within the model's allowed
        range and returns a numpy array if successfull.
        """
        array = np.asarray(wavelength)
        lo, hi = self.wavelengthRange
        if array.min() < lo or array.max() > hi:
            raise ValueError(
                f"Wavelength outside the model's allowed range of ({lo}-{hi})nm!"
            )
        return array

    def checkCosTheta(self, cos_theta: ArrayLike) -> NDArray:
        """
        Checks whether the given cosines are in the valid range [-1,1] and
        returns them as numpy array if successfull.
        """
        array = np.asarray(cos_theta)
        if array.min() < -1 or array.max() > 1:
            raise ValueError("Cosines outside the value range of [-1,1]!")
        return array

    def checkEta(self, eta: ArrayLike) -> NDArray:
        """
        Checks whether the given random numbers are in the valie range [0,1]
        and returns them as numpy array if successfull.
        """
        array = np.asarray(eta)
        if array.min() < 0 or array.max() > 1:
            raise ValueError("Eta outside the valid range of [0,1]!")
        return array

    def refractive_index(self, wavelength: ArrayLike) -> NDArray | None:
        """
        Calculates the refractive index for the given wavelengths.
        Returns None if not defined.
        """
        return None

    def group_velocity(self, wavelength: ArrayLike) -> NDArray | None:
        """
        Calculates the group velocity for the given wavelengths.
        Returns None if not defined.
        """
        return None

    def absorption_coef(self, wavelength: ArrayLike) -> NDArray | None:
        """
        Returns the absorption coefficient for the given wavelengths.
        Returns None if not defined.
        """
        return None

    def scattering_coef(self, wavelength: ArrayLike) -> NDArray | None:
        """
        Returns the scattering coefficient for the given wavelengths.
        Returns None if not defined.
        """
        return None

    def log_phase_function(self, cos_theta: ArrayLike) -> NDArray | None:
        """
        Evaluates the log phase functions for the given values of cos theta
        with theta being the angle between incoming and outgoing ray after
        scattering.
        Returns None if not defined.
        """
        return None

    def phase_sampling(self, eta: ArrayLike) -> NDArray | None:
        """
        Returns samples of cos(theta) using the uniform unit random numbers eta
        which follows the phase function as underlying distribution.
        Returns None if not defined.
        """
        return None

    def phase_m12(self, cos_theta: ArrayLike) -> ArrayLike | None:
        """
        Returns samples of the m12 element of the phase matrix for the given
        values of cos theta with theta being the angle between incoming and
        outgoing ray after scattering.
        Returns None if not defined.
        """
        return None

    def phase_m22(self, cos_theta: ArrayLike) -> ArrayLike | None:
        """
        Returns samples of the m22 element of the phase matrix for the given
        values of cos theta with theta being the angle between incoming and
        outgoing ray after scattering.
        Returns None if not defined.
        """
        return None

    def phase_m33(self, cos_theta: ArrayLike) -> ArrayLike | None:
        """
        Returns samples of the m33 element of the phase matrix for the given
        values of cos theta with theta being the angle between incoming and
        outgoing ray after scattering.
        Returns None if not defined.
        """
        return None

    def phase_m34(self, cos_theta: ArrayLike) -> ArrayLike | None:
        """
        Returns samples of the m34 element of the phase matrix for the given
        values of cos theta with theta being the angle between incoming and
        outgoing ray after scattering.
        Returns None if not defined.
        """
        return None

    def createMedium(
        self,
        *,
        wavelengthRange: tuple[float, float] | None = None,
        numLambda: int = 1024,
        numTheta: int = 1024,
        name: str | None = None,
    ) -> Medium:
        """
        Creates a medium using this model by sampling its properties.

        Parameters
        ----------
        wavelengthRange: (float, float) | None, default=None
            Range of wavelength to sample. If `None`, uses the model's full
            valid range.
        numLambda: int, default=1024
            Number of samples to take from the wavelength range
        numTheta: int, default=1024
            Number of samples to take from the phase and its sample function
        name: str | None, default=None
            Name of the medium. If `None`, uses the model's name.
        """
        if wavelengthRange is None:
            wavelengthRange = self.wavelengthRange
        l = np.linspace(*wavelengthRange, numLambda)
        t = np.linspace(-1.0, 1.0, numTheta)  # cos(theta)
        e = np.linspace(0.0, 1.0, numTheta)
        return Medium(
            name if name is not None else self.name,
            *wavelengthRange,
            refractive_index=self.refractive_index(l),
            group_velocity=self.group_velocity(l),
            absorption_coef=self.absorption_coef(l),
            scattering_coef=self.scattering_coef(l),
            log_phase_function=self.log_phase_function(t),
            phase_sampling=self.phase_sampling(e),
            phase_m12=self.phase_m12(t),
            phase_m22=self.phase_m22(t),
            phase_m33=self.phase_m33(t),
            phase_m34=self.phase_m34(t),
        )


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

    def phase_sampling(self, eta: ArrayLike) -> NDArray:
        eta = self.checkEta(eta)
        if self._phaseSampler is not None:
            sampler = self._phaseSampler
        else:
            if "phase_function" in dir(self):
                dist = NumericalPhaseSamplingMixin._DistWrapper(self)
            elif self.log_phase_function(0.0) is not None:
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

    def refractive_index(self, wavelength: ArrayLike) -> NDArray:
        """Calculates the refractive index for the given wavelengths"""
        L2 = np.square(self.checkWavelength(wavelength))
        # L2 = np.square(wavelength)
        S1 = self.B1 * L2 / (L2 - self.C1)
        S2 = self.B2 * L2 / (L2 - self.C2)
        S3 = self.B3 * L2 / (L2 - self.C3)
        return np.sqrt(1.0 + S1 + S2 + S3)

    def group_velocity(self, wavelength: ArrayLike) -> NDArray:
        """
        Calculates the group velocity in for the given wavelengths
        """
        L = self.checkWavelength(wavelength)
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

    def absorption_coef(self, wavelength: ArrayLike) -> NDArray:
        """Returns the absorption coefficient for the given wavelengths"""
        assert BK7Model.TransmissionTable is not None
        # we can transform the transmission measurements to absorption
        # coefficients via the Beer-Lambert law. Unfortunately, they two
        # measurements disagree a lot. So we take the average weighted by
        # the probe thickness, as thicker ones should give a better result
        # To avoid taking the average with inf, we actually take the average
        # of the absorption lengths, i.e. inf -> 0
        wavelength = self.checkWavelength(wavelength)

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
        self._delta = depolarizationRatio

    @property
    def depolarizationRatio(self) -> float:
        """Depolarization ratio of the medium"""
        return self._delta

    @depolarizationRatio.setter
    def depolarizationRatio(self, value: float) -> None:
        self._delta = value

    def phase_function(self, cos_theta: ArrayLike) -> NDArray:
        #                    1 - delta
        # beta(theta) ~ 1 + ----------- cos^2(theta)
        #                    1 + delta
        cos_theta = self.checkCosTheta(cos_theta)
        delta = self.depolarizationRatio
        phase = 1 + (1 - delta) / (1 + delta) * cos_theta**2
        I = 8 * np.pi / 3 * (2 + delta) / (1 + delta)  # integral phase from -1 to 1
        return phase / I

    def log_phase_function(self, cos_theta: ArrayLike) -> NDArray:
        return np.log(self.phase_function(cos_theta))

    def phase_m12(self, cos_theta: ArrayLike) -> NDArray:
        # M_11 = beta_90 * 1 + (1-d)/(1+d)cos^2
        # M_12 = beta_90 * -(1-d)/(1+d)sin^2
        #            M_12       -(1-d) sin^2
        # -> m_12 = ------ = ------------------
        #            M_11     1+d + (1-d)cos^2
        d = self.depolarizationRatio
        u = self.checkCosTheta(cos_theta)
        return -(1 - d) * (1 - u**2) / (1 + d + (1 - d) * u**2)

    def phase_m22(self, cos_theta: ArrayLike) -> NDArray:
        # M_22 = M_11 -> m_22 = 1
        return np.ones_like(cos_theta)

    def phase_m33(self, cos_theta: ArrayLike) -> NDArray:
        # M_11 = beta_90 * 1 + (1-d)/(1+d)cos^2
        # M_33 = beta_90 * 2(1-d)/(1+d)cos
        #            M_33        2(1-d)cos
        # -> m_33 = ------ = ------------------
        #            M_11     1+d + (1-d)cos^2
        d = self.depolarizationRatio
        u = self.checkCosTheta(cos_theta)
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
        self._temp = temperature
        self._betaT = betaT

    @property
    def temperature(self) -> float:
        """Temperature of the media in °C"""
        return self._temp

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temp = value

    @property
    def betaT(self) -> float:
        """Isothermal compressibility of the medium in m^3/J"""
        return self._betaT

    @betaT.setter
    def betaT(self, value: float) -> None:
        self._betaT = value

    def scattering_coef(self, wavelength: ArrayLike) -> NDArray:
        #      4pi^3 * k_B*T * beta_T                  2 + delta
        # b = ------------------------ * LL^2 * f_C * -----------
        #            2 * lam^4                         1 + delta
        lam = u.convert(self.checkWavelength(wavelength), u.m)
        T = self.temperature
        betaT = self.betaT
        delta = self.depolarizationRatio
        E = scipy.constants.Boltzmann * (T + 273.15)
        # Cabannes factor; See [ZH21] Eq. 4
        fc = (6 + 6 * delta) / (6 - 7 * delta)
        fc *= (2 + delta) / (1 + delta)  # from integration
        # Lorentz-Lorenz; See [ZH21] Eq. 9
        n = self.refractive_index(wavelength)
        if n is None:
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

    def log_phase_function(self, cos_theta: ArrayLike) -> NDArray:
        """
        Evaluates the log phase function for the given angles as cos(theta).
        Normalized with respect to unit sphere.
        """
        cos_theta = self.checkCosTheta(cos_theta)
        return np.log(
            (1.0 - self.g**2)
            / np.power(1.0 + self.g**2 - 2 * self.g * cos_theta, 1.5)
            / (4.0 * np.pi)
        )

    def phase_sampling(self, eta: ArrayLike) -> NDArray:
        """
        Samples the phase function using provided unit random numbers eta.
        Returns the cosine of the sampled angle.

        See Zhang, J.: On Sampling of Scattering Phase Functions, 2019
        """
        eta = self.checkEta(eta)
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

    def log_phase_function(self, cos_theta: ArrayLike) -> NDArray:
        """Evaluates the log phase function for the given angles mu = cos(theta)"""
        # phase functions becomes singular at cos_theta = 1.0
        # clip close before (1 float ulp)
        cos_theta = self.checkCosTheta(cos_theta)
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
    ng: float, default=1.0
        Group index (group velocity = c/ng)
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
        ng: float = 1.0,
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
        self.ng = ng
        self.mu_a = mu_a
        self.mu_s = mu_s

    @property
    def n(self) -> float:
        """Refractive index"""
        return self._n

    @n.setter
    def n(self, value: float) -> None:
        self._n = value

    @property
    def ng(self) -> float:
        """Group index (group velocity = c/ng)"""
        return self._ng

    @ng.setter
    def ng(self, value: float) -> None:
        self._ng = value

    @property
    def mu_a(self) -> float:
        """Absorption coefficient"""
        return self._mu_a

    @mu_a.setter
    def mu_a(self, value: float) -> None:
        self._mu_a = value

    @property
    def mu_s(self) -> float:
        """Scattering coefficient"""
        return self._mu_s

    @mu_s.setter
    def mu_s(self, value: float) -> None:
        self._mu_s = value

    def refractive_index(self, wavelength: ArrayLike) -> NDArray:
        if not np.min(wavelength) >= 0.0:
            raise ValueError("wavelength must be positive!")
        return np.ones_like(wavelength) * self.n

    def group_velocity(self, wavelength: ArrayLike) -> NDArray:
        if not np.min(wavelength) >= 0.0:
            raise ValueError("wavelength must be positive!")
        return np.ones_like(wavelength) / self.ng * u.c

    def absorption_coef(self, wavelength: ArrayLike) -> NDArray:
        if not np.min(wavelength) >= 0.0:
            raise ValueError("wavelength must be positive!")
        return np.ones_like(wavelength) * self.mu_a

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

    def refractive_index(self, wavelength: ArrayLike) -> NDArray:
        # formula expects wavelengths in micrometers -> convert
        L = u.convert(self.checkWavelength(wavelength), u.um)
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

    def group_velocity(self, wavelength: ArrayLike) -> NDArray:
        # formula expects wavelengths in micrometers -> convert
        L = u.convert(self.checkWavelength(wavelength), u.um)
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

    def absorption_coef(self, wavelength: ArrayLike) -> NDArray:
        # linear interpolate table
        L = self.checkWavelength(wavelength)
        assert PureWaterModel._MU_A_SPLINE is not None  # make the linter happy...
        return PureWaterModel._MU_A_SPLINE(L)

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

    def scattering_coef(self, wavelength: ArrayLike) -> NDArray:
        # Code based on matlab code from [ZH21]
        Kbz = 1.3806503e-23  #  Boltzmann constant
        Tk = self.temperature + 273.15  #  Absolute temperature
        lam = self.checkWavelength(wavelength)
        delta = self.depolarizationRatio
        nsw = self.refractive_index(wavelength)
        dnswds = self._dn_dS(wavelength)
        nsw2 = nsw**2
        DFRI = (nsw2 - 1) * (1 + 2 / 3 * (nsw2 + 2) * (nsw / 3 - 1 / 3 / nsw) ** 2)
        SFRI = 2 * nsw * dnswds
        gp = self._dg_dp()
        gpp = self._dg_dp2()
        beta90 = np.pi**2 * Kbz / 2 * ((lam * 1e-9) ** (-4)) * Tk
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

    def absorption_coef(self, wavelength: ArrayLike) -> NDArray:
        lam = self.checkWavelength(wavelength)
        table = SmithNaturalWaterMeasurements._TABLE
        assert table is not None
        return np.interp(lam, table[:, 0], table[:, 1])

    def scattering_coef(self, wavelength: ArrayLike) -> NDArray:
        lam = self.checkWavelength(wavelength)
        table = SmithNaturalWaterMeasurements._TABLE
        assert table is not None
        return np.interp(lam, table[:, 0], table[:, 2])


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

    @property
    def p90(self) -> float:
        """Degree of polarization at 90°"""
        return self._p90

    @p90.setter
    def p90(self, value: float) -> None:
        self._p90 = value

    @property
    def theta0(self) -> float:
        """Shift in angle"""
        return self._theta0

    @theta0.setter
    def theta0(self, value) -> None:
        self._theta0 = value

    @property
    def alpha(self) -> float:
        """multipole scatter slope"""
        return self._alpha

    @alpha.setter
    def alpha(self, value) -> None:
        self._alpha = value

    @property
    def xi(self) -> float:
        """multipole scatter scale"""
        return self._xi

    @xi.setter
    def xi(self, value: float) -> None:
        self._xi = value

    def phase_m12(self, cos_theta: ArrayLike) -> NDArray:
        """m12 element of the phase matrix"""
        cos_theta = self.checkCosTheta(cos_theta)
        cos_theta_sq = np.square(cos_theta)
        sin_theta_sq = 1.0 - cos_theta_sq
        return -self.p90 * sin_theta_sq / (1.0 + self.p90 * cos_theta_sq)

    def phase_m22(self, cos_theta: ArrayLike) -> NDArray:
        """m22 element of the phase matrix"""
        cos_theta = self.checkCosTheta(cos_theta)
        theta = np.arccos(cos_theta)
        z = theta - self.theta0
        cos_z_sq = np.square(np.cos(z))
        e = self.xi * np.exp(-self.alpha * theta)
        return (self.p90 * (1.0 + cos_z_sq) + e) / (1.0 + self.p90 * cos_z_sq + e)

    def phase_m33(self, cos_theta: ArrayLike) -> NDArray:
        """m33 element of the phase matrix"""
        cos_theta = self.checkCosTheta(cos_theta)
        cos_theta = np.asarray(cos_theta)
        theta = np.arccos(cos_theta)
        ct_sq = np.square(cos_theta)
        e = self.xi * np.exp(-self.alpha * theta)
        return (2 * self.p90 * cos_theta + e) / (1.0 + self.p90 * ct_sq + e)
