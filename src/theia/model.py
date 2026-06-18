from __future__ import annotations

import warnings

import numpy as np

import scipy.constants
from scipy.interpolate import CubicSpline
from scipy.stats.sampling import NumericalInversePolynomial

from types import SimpleNamespace

from theia.lookup import Interpolation, Table, createTableFromFunction
from theia.material import Material, MaterialFlags, Medium
from theia.property import FloatProperty, TableProperty, Property
from theia.surface import SurfaceModel
from theia.volume import Transparent, VolumeModel
from theia.util import classproperty, loadCSV, intersectRange
import theia.units as u

from collections.abc import Callable
from numpy.typing import ArrayLike, NDArray
from typing import Any, Generic, TypeVar, overload
from typing_extensions import Self

__all__ = [
    "angular_property",
    "numerical_quantile_property",
    "numerical_sampler_property",
    "optical_constant",
    "optical_property",
    "sampler_property",
    "BK7Model",
    "DispersionFreeMedium",
    "FournierForandPhaseFunction",
    "HenyeyGreensteinPhaseFunction",
    "KokhanovskyOceanWaterPhaseMatrix",
    "OpticalModel",
    "PureWaterModel",
    "RayleighScatteringModel",
    "RayleighScatteringPhaseFunction",
    "SellmeierEquation",
    "SmithNaturalWaterMeasurements",
    "WaterBaseModel",
]


def __dir__():
    return __all__


class _optical_constant:
    """Class marking optical constants in models. See `optical_constant`"""

    def __init__(
        self,
        fn: Callable[[Any], float],
        *,
        name: str | None = None,
    ) -> None:
        self.fn = fn
        self.name = name

    def __set_name__(self, owner, name: str) -> None:
        if self.name is None:
            self.name = name

    @overload
    def __get__(self, obj: None, owner=None) -> Self: ...
    @overload
    def __get__(self, obj: Any, owner=None) -> float: ...
    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return self.fn(obj)

    def createProperty(
        self,
        obj,
        defaultRange: tuple[float, float],
        numSamples: int,
    ) -> Property:
        return FloatProperty(self.fn(obj))


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


class _optical_property(Generic[_R]):
    """Class marking optical properties in models. See `optical_property`"""

    def __init__(
        self,
        fn: Callable[..., _R],
        *,
        name: str | None = None,
        range: list[tuple[float, float]] | tuple[float, float] | None = None,
        const: bool = False,
        interpolation: Interpolation = "linear",
    ) -> None:
        self.fn = fn
        self.name = name
        if isinstance(range, tuple):
            range = [range]
        self.range = range
        self.const = const
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
    ) -> Property:
        """Creates a `Property` from this medium property"""
        range = [defaultRange] if self.range is None else self.range
        if self.const:
            x = (r[0] for r in range)
            y = self.fn(obj, *x)
            table = Table(np.asarray(y).ravel())
        else:
            xi = [(*r, numSamples) for r in range]
            fn = lambda *xi: self.fn(obj, *xi)  # bind object as self
            table = createTableFromFunction(fn, *xi, interpolation=self.interpolation)
        return TableProperty(table)


class _DistWrapper:
    """Wrapper for pdf to be compatible with `NumericalInversePolynomial`"""

    def __init__(self, fn) -> None:
        self.fn = fn

    def pdf(self, x: float) -> float:
        # we cannot be sure if we get an array or a scalar back
        return np.asarray(self.fn(x)).item()


class _LogDistWrapper:
    """Wrapper for log pdf to be compatible with `NumericalInversePolynomial`"""

    def __init__(self, fn) -> None:
        self.fn = fn

    def logpdf(self, x: float) -> float:
        return np.asarray(self.fn(x)).item()


def _createSampler(
    obj: Any, prop: _optical_property[_R], log: bool
) -> NumericalInversePolynomial:
    """Creates a `NumericalInversePolynomial` for the given optical property"""
    if prop.range is None:
        raise ValueError("A range must be specified!")
    if len(prop.range) != 1:
        raise ValueError("Can only numerically inverse 1D functions")

    fn = lambda x: prop.fn(obj, x)
    if log:
        dist = _LogDistWrapper(fn)
    else:
        dist = _DistWrapper(fn)
    return NumericalInversePolynomial(dist, domain=prop.range[0])


class numerical_sampler_property(Generic[_R]):
    """
    Creates a sampling function from the given optical property by
    numerical inverting its CDF. The sampler will also be made available as
    optical property.

    Parameters
    ----------
    prop: optical_property
        Optical property to sample
    name: str | None, default=None
        Name of the property. If `None`, uses the function's name
    log: bool, default=False
        Whether the underlying distribution is in log space
    interpolation: Interpolation, default="linear"
        Interpolation to use when creating look up table from the sampled
        values. Ignored on properties.
    """

    def __init__(
        self,
        prop: _optical_property[_R],
        *,
        name: str | None = None,
        log: bool = False,
        interpolation: Interpolation = "linear",
    ) -> None:
        if prop.range is None:
            raise ValueError("A range must be specified!")
        if len(prop.range) != 1:
            raise ValueError("Can only numerically inverse 1D functions")
        self.domain = prop.range[0]

        self.prop = prop
        self.log = log
        self.name = name
        self.interpolation: Interpolation = interpolation

    def __set_name__(self, owner, name: str) -> None:
        if self.name is None:
            self.name = name

    @overload
    def __get__(self, obj: None, owner=None) -> Self: ...

    @overload
    def __get__(self, obj: Any, owner=None) -> Callable[[ArrayLike], ArrayLike]: ...

    # TODO: Fix type annotation

    def __get__(self, obj, owner=None):
        if obj is None:
            # accessed from class
            return self
        # TODO: might want to cache the sampler...
        sampler = _createSampler(obj, self.prop, self.log)
        return lambda x: sampler.ppf(x)

    def createProperty(
        self,
        obj,
        defaultRange: tuple[float, float],
        numSamples: int,
    ) -> Property:
        sampler = _createSampler(obj, self.prop, self.log)
        fn = lambda x: sampler.ppf(x)
        table = createTableFromFunction(
            fn, (0.0, 1.0, numSamples), interpolation=self.interpolation
        )
        return TableProperty(table)


class numerical_quantile_property(Generic[_R]):
    """
    Creates a function numerically calculating the quantiles of the given
    optical property and making it available as optical property itself.

    Parameters
    ----------
    prop: optical_property
        Optical property to calculate quantiles from
    name: str | None, default=None
        Name of the property. If `None`, uses the function's name.
    log: bool, default=False
        Whether the underlying distribution is in log space.
    interpolation: Interpolation, default="linear"
        Interpolation to use when creating the look up table from sampled
        sampled values.
    """

    def __init__(
        self,
        prop: _optical_property[_R],
        *,
        name: str | None = None,
        log: bool = False,
        interpolation: Interpolation = "linear",
    ) -> None:
        if prop.range is None:
            raise ValueError("A range must be specified!")
        if len(prop.range) != 1:
            raise ValueError("Can only numerically inverse 1D functions")
        self.domain = prop.range[0]

        self.prop = prop
        self.name = name
        self.log = log
        self.interpolation = interpolation

    def __set_name__(self, owner, name: str) -> None:
        if self.name is None:
            self.name = name

    @overload
    def __get__(self, obj: None, owner=None) -> Self: ...
    @overload
    def __get__(self, obj: Any, owner=None) -> Callable[[ArrayLike], ArrayLike]: ...

    def __get__(self, obj, owner=None):
        if obj is None:
            # accessed from class
            return self
        sampler = _createSampler(obj, self.prop, self.log)
        return lambda x: sampler.cdf(x)

    def createProperty(
        self,
        obj,
        defaultRange: tuple[float, float],
        numSamples: int,
    ) -> Property:
        sampler = _createSampler(obj, self.prop, self.log)
        fn = lambda x: sampler.cdf(x)
        table = createTableFromFunction(
            fn, (*self.domain, numSamples), interpolation=self.interpolation
        )
        return TableProperty(table)


_optical_property_types: tuple[type, ...] = (
    _optical_constant,
    _optical_property,
    numerical_quantile_property,
    numerical_sampler_property,
)


@overload
def optical_constant(fn: Callable[[Any], float]) -> _optical_constant: ...


@overload
def optical_constant(
    *, name: None | str = ...
) -> Callable[[Callable[[Any], float]], _optical_constant]: ...


def optical_constant(
    fn: Callable[[Any], float] | None = None,
    *,
    name: str | None = None,
) -> Callable[[Callable[[Any], float]], _optical_constant] | _optical_constant:
    """
    Can be used in classes derived from `OpticalModel` to mark functions as
    optical constants. These will be used during property creation to populate
    the corresponding `FloatProperty`.

    Parameters
    ----------
    fn:
        Function to decorate
    name: str | None, default=None
        Name of the constant. If `None`, uses the function's name

    Examples
    --------
    >>> class Model(OpticalModel):
    ...     @optical_constant
    ...     def temperature(self) -> float:
    ...         return 300.0
    ...     @optical_constant(name="thickness")
    ...     def d(self) -> float:
    ...         return 0.4
    """
    # need this trick to allow arguments
    if fn is not None:
        return _optical_constant(fn)
    else:
        return lambda func: _optical_constant(func, name=name)


@overload
def optical_property(fn: Callable[..., _R]) -> _optical_property[_R]: ...


@overload
def optical_property(
    *,
    name: None | str = ...,
    range: list[tuple[float, float]] | tuple[float, float] | None = ...,
    const: bool = ...,
    interpolation: Interpolation = ...,
) -> Callable[[Callable[..., _R]], _optical_property[_R]]: ...


def optical_property(
    fn: Callable[..., _R] | None = None,
    *,
    name: str | None = None,
    range: list[tuple[float, float]] | tuple[float, float] | None = None,
    const: bool = False,
    interpolation: Interpolation = "linear",
) -> Callable[[Callable[..., _R]], _optical_property[_R]] | _optical_property[_R]:
    """
    Can be used in classes derived from `OpticalModel` to mark functions or
    properties as optical properties. These will be used during property creation
    to populate the corresponding `PropertyTable`.

    Functions will be sampled to create `TableProperty`. If they are marked with
    `property` a `FloatProperty` is created instead and no sampling happens.

    Parameters
    ----------
    fn:
        Function to decorate
    name: str | None, default=None
        Name of the property. If `None`, uses the function's name
    range: ((float, float), ...) | None, default=None
        Range over which to sample the function for each axis. If `None`, uses
        the wavelength range of the model. Ignored if decorated with `property`.
    const: bool, default=False
        Marks property as constant and only a single sample will be stored in
        the created `Table` resulting in faster look ups.
        Ignored if decorated with `property`.
    interpolation: Interpolation, default="linear"
        Interpolation to use when creating look up table from the sampled
        values. Ignored on properties.

    Examples
    --------
    >>> class Model(OpticalModel):
    ...     @optical_property
    ...     def refractive_index(self, wavelength):
    ...         return 1.41 * np.ones_like(wavelength)
    ...     @optical_property(range=(-1.,1.), const=True)
    ...     def log_phase_function(self, cos_theta):
    ...         return np.ones_like(cos_theta) / (4.0 * np.pi)
    """
    # need this trick to allow arguments
    if fn is not None:
        # decorated without parameters
        return _optical_property(fn)
    else:
        return lambda func: _optical_property(
            func, name=name, range=range, const=const, interpolation=interpolation
        )


@overload
def angular_property(fn: Callable[..., _R]) -> _optical_property[_R]: ...


@overload
def angular_property(
    *,
    name: None | str = ...,
    const: bool = ...,
    interpolation: Interpolation = ...,
) -> Callable[[Callable[..., _R]], _optical_property[_R]]: ...


def angular_property(
    fn: Callable[..., _R] | None = None,
    *,
    name: str | None = None,
    const: bool = False,
    interpolation: Interpolation = "linear",
) -> Callable[[Callable[..., _R]], _optical_property[_R]] | _optical_property[_R]:
    if fn is not None:
        return _optical_property(fn, range=(-1.0, 1.0))
    else:
        return lambda func: _optical_property(
            func, name=name, range=(-1.0, 1.0), const=const, interpolation=interpolation
        )


@overload
def sampler_property(fn: Callable[..., _R]) -> _optical_property[_R]: ...


@overload
def sampler_property(
    *,
    name: None | str = ...,
    const: bool = ...,
    interpolation: Interpolation = ...,
) -> Callable[[Callable[..., _R]], _optical_property[_R]]: ...


def sampler_property(
    fn: Callable[..., _R] | None = None,
    *,
    name: str | None = None,
    const: bool = False,
    interpolation: Interpolation = "linear",
) -> Callable[[Callable[..., _R]], _optical_property[_R]] | _optical_property[_R]:
    if fn is not None:
        return _optical_property(fn, range=(0.0, 1.0))
    else:
        return lambda func: _optical_property(
            func, name=name, range=(0.0, 1.0), const=const, interpolation=interpolation
        )


class OpticalModel:
    """
    Base class for models of optical properties. Implements methods to create
    `Medium` and `Material` by sampling the an

    Parameters
    ----------
    name: str, default="noname"
        Name of the model. Will be used as default name when creating material
        and media.
    wavelengthRange: (float, float), default=(100.0, 1500.0) nm
        Range of wavelength in nm for which the model is valid. Sampling of
        optical properties will use this range where applicable.
    """

    def __init__(
        self,
        *args,
        name: str = "noname",
        wavelengthRange: tuple[float, float] = (100.0, 10000.0) * u.nm,
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
            if isinstance(prop, _optical_property):
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

    def createProperties(
        self,
        *,
        wavelengthRange: tuple[float, float] | None = None,
        numSamples: int = 1024,
    ) -> dict[str, Property]:
        if wavelengthRange is None:
            wavelengthRange = self.wavelengthRange

        # collect medium properties
        props = {}
        for atr in dir(self):
            if atr.startswith("_"):
                continue
            # fetch properties from class object to sidestep descriptor
            prop = getattr(self.__class__, atr, None)
            if prop is None:
                continue
            if isinstance(prop, _optical_property_types):
                props[prop.name] = prop.createProperty(
                    self, wavelengthRange, numSamples
                )
        return props

    def createMaterial(
        self,
        inside: Medium | str | None,
        outside: Medium | str | None,
        physicModel: SurfaceModel,
        *,
        flags: (
            tuple[MaterialFlags | str, MaterialFlags | str] | MaterialFlags | str
        ) = MaterialFlags(0),
        wavelengthRange: tuple[float, float] | None = None,
        numSamples: int = 1024,
        name: str | None = None,
        extraProperties: dict[str, Property] = {},
    ) -> Material:
        if wavelengthRange is None:
            wavelengthRange = self.wavelengthRange
        if name is None:
            name = self.name

        props = self.createProperties(
            wavelengthRange=wavelengthRange,
            numSamples=numSamples,
        )
        props.update(extraProperties)

        return Material(
            name, inside, outside, physicModel, flags=flags, properties=props
        )

    def createMedium(
        self,
        *,
        wavelengthRange: tuple[float, float] | None = None,
        numSamples: int = 1024,
        name: str | None = None,
        physicModel: VolumeModel = Transparent(),
        extraProperties: dict[str, Property] = {},
    ) -> Medium:
        if wavelengthRange is None:
            wavelengthRange = self.wavelengthRange
        if name is None:
            name = self.name

        props = self.createProperties(
            wavelengthRange=wavelengthRange,
            numSamples=numSamples,
        )
        props.update(extraProperties)

        return Medium(name, wavelengthRange, props, physicModel)


#################################### MODELS ####################################


class SellmeierEquation(OpticalModel):
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

    @optical_property
    def refractive_index(self, wavelength: NDArray) -> NDArray:
        """Calculates the refractive index for the given wavelengths"""
        L2 = np.square(wavelength)
        # L2 = np.square(wavelength)
        S1 = self.B1 * L2 / (L2 - self.C1)
        S2 = self.B2 * L2 / (L2 - self.C2)
        S3 = self.B3 * L2 / (L2 - self.C3)
        return np.sqrt(1.0 + S1 + S2 + S3)

    @optical_property
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            1.03961212,  # B1
            0.231792344,  # B2
            1.010469450,  # B3
            0.00600069867e6,  # C1 [nm^2]
            0.0200179144e6,  # C2 [nm^2]
            103.5606530e6,  # C3 [nm^2]
            *args,
            name="bk7",
            **kwargs,
        )
        self.restrictWavelengthRange((200.0, 1060.0) * u.nm)
        # lazily load data
        if BK7Model.TransmissionTable is None:
            BK7Model.TransmissionTable = loadCSV("bk7_transmission.csv", skiprows=2)

    @optical_property
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


class RayleighScatteringPhaseFunction(OpticalModel):
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

    def phase_function(self, cos_theta: NDArray) -> NDArray:
        #                    1 - delta
        # beta(theta) ~ 1 + ----------- cos^2(theta)
        #                    1 + delta
        delta = self.depolarizationRatio
        phase = 1 + (1 - delta) / (1 + delta) * cos_theta**2
        I = 8 * np.pi / 3 * (2 + delta) / (1 + delta)  # integral phase from -1 to 1
        return phase / I

    @angular_property
    def log_phase_function(self, cos_theta: NDArray) -> NDArray:
        return np.log(self.phase_function(cos_theta))

    phase_sampling = numerical_sampler_property(log_phase_function, log=True)

    @angular_property
    def phase_m12(self, cos_theta: NDArray) -> NDArray:
        # M_11 = beta_90 * 1 + (1-d)/(1+d)cos^2
        # M_12 = beta_90 * -(1-d)/(1+d)sin^2
        #            M_12       -(1-d) sin^2
        # -> m_12 = ------ = ------------------
        #            M_11     1+d + (1-d)cos^2
        d = self.depolarizationRatio
        u = cos_theta
        return -(1 - d) * (1 - u**2) / (1 + d + (1 - d) * u**2)

    @angular_property
    def phase_m22(self, cos_theta: NDArray) -> NDArray:
        # M_22 = M_11 -> m_22 = 1
        return np.ones_like(cos_theta)

    @angular_property
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

    @optical_property
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


class HenyeyGreensteinPhaseFunction(OpticalModel):
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

    @angular_property
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

    @sampler_property
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


class FournierForandPhaseFunction(OpticalModel):
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

    @property
    def mu(self) -> float:
        """The slope of the particle size distribution log N(r) ~ -mu*log(r)"""
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        self._mu = value

    @angular_property
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

    phase_sampling = numerical_sampler_property(log_phase_function, log=True)


class DispersionFreeMedium(OpticalModel):
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

    @optical_property(const=True)
    def refractive_index(self, wavelength: ArrayLike) -> NDArray:
        if not np.min(wavelength) >= 0.0:
            raise ValueError("wavelength must be positive!")
        return np.ones_like(wavelength) * self.n

    @optical_property(const=True)
    def group_velocity(self, wavelength: ArrayLike) -> NDArray:
        if not np.min(wavelength) >= 0.0:
            raise ValueError("wavelength must be positive!")
        return np.ones_like(wavelength) / self.ng * u.c

    @optical_property(const=True)
    def absorption_coef(self, wavelength: ArrayLike) -> NDArray:
        if not np.min(wavelength) >= 0.0:
            raise ValueError("wavelength must be positive!")
        return np.ones_like(wavelength) * self.mu_a

    @optical_property(const=True)
    def scattering_coef(self, wavelength: ArrayLike) -> NDArray:
        if not np.min(wavelength) >= 0.0:
            raise ValueError("wavelength must be positive!")
        return np.ones_like(wavelength) * self.mu_s


class WaterBaseModel(OpticalModel):
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

    @optical_property
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

    @optical_property
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

    @optical_property
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

    @optical_property
    def scattering_coef(self, wavelength: NDArray) -> NDArray:
        # Code based on matlab code from [ZH21]
        Kbz = 1.3806503e-23  #  Boltzmann constant
        Tk = self.temperature + 273.15  #  Absolute temperature
        lam = wavelength * 1e-9
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


class SmithNaturalWaterMeasurements(OpticalModel):
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

    @optical_property
    def absorption_coef(self, wavelength: NDArray) -> NDArray:
        table = SmithNaturalWaterMeasurements._TABLE
        assert table is not None
        return np.interp(wavelength, table[:, 0], table[:, 1])

    @optical_property
    def scattering_coef(self, wavelength: NDArray) -> NDArray:
        table = SmithNaturalWaterMeasurements._TABLE
        assert table is not None
        return np.interp(wavelength, table[:, 0], table[:, 2])


class KokhanovskyOceanWaterPhaseMatrix(OpticalModel):
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

    @angular_property
    def phase_m12(self, cos_theta: ArrayLike) -> NDArray:
        """m12 element of the phase matrix"""
        cos_theta_sq = np.square(cos_theta)
        sin_theta_sq = 1.0 - cos_theta_sq
        return -self.p90 * sin_theta_sq / (1.0 + self.p90 * cos_theta_sq)

    @angular_property
    def phase_m22(self, cos_theta: ArrayLike) -> NDArray:
        """m22 element of the phase matrix"""
        theta = np.arccos(cos_theta)
        z = theta - self.theta0
        cos_z_sq = np.square(np.cos(z))
        e = self.xi * np.exp(-self.alpha * theta)
        return (self.p90 * (1.0 + cos_z_sq) + e) / (1.0 + self.p90 * cos_z_sq + e)

    @angular_property
    def phase_m33(self, cos_theta: ArrayLike) -> NDArray:
        """m33 element of the phase matrix"""
        cos_theta = np.asarray(cos_theta)
        theta = np.arccos(cos_theta)
        ct_sq = np.square(cos_theta)
        e = self.xi * np.exp(-self.alpha * theta)
        return (2 * self.p90 * cos_theta + e) / (1.0 + self.p90 * ct_sq + e)
