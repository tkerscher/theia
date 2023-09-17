import importlib.resources
import numpy as np
import numpy.typing as npt
import hephaistos as hp
import theia.lookup
import warnings
from collections.abc import Iterable
from ctypes import Structure, c_float, c_uint64, addressof, memmove, sizeof
from scipy.constants import speed_of_light
from scipy.interpolate import CubicSpline
from typing import Union


#################################### COMMON ####################################


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
        Table of group velocity in units of c as function of wavelength.
        None defaults to a constant value of 1.0 (c).
    absorption_coef: ArrayLike | None, default = None
        Table of absorption coefficient in units of 1/m as function of wavelength.
        None defaults to a constant value of 0.0.
    scattering_coef: ArrayLike | None, default = None
        Table of scattering coefficient in units of 1/m as function of wavelength.
        None defaults to a constant value of 0.0.
    phase_function: ArrayLike | None, default = None
        Table of scattering phase function as a function of the cosine of the
        angle between incoming and outgoing ray in radians over the range [0,1].
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
            ("phase", c_uint64),  # Table1D
            ("phase_sampling", c_uint64),  # Table1D
        ]

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
        phase_function: Union[npt.ArrayLike, None] = None,
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
        self.phase_function = phase_function
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
        Table containing values of the group velocity in units of c as a
        function of wavelength sampled at equidistant points on the range
        defined by lambda min/max.
        If None, a constant value of 1.0 is assumed.
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
    def phase_function(self) -> Union[npt.ArrayLike, None]:
        """
        Table of scattering phase function as a function of the cosine of the
        angle between incoming and outgoing ray in radians over the range [0,1].
        None defaults to a constant value of 1, i.e. uniform scattering.
        """
        return self._phase_function

    @phase_function.setter
    def phase_function(self, value: Union[npt.ArrayLike, None]) -> None:
        self._phase_function = value

    @phase_function.deleter
    def phase_function(self) -> None:
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

    @property
    def byte_size(self) -> int:
        """
        Amount of bytes needed to store this medium alongside its referenced
        properties on the GPU.
        """
        size = 0
        # first sum up the size we need to store the tables
        size += theia.lookup.getTableSize(self.refractive_index)
        size += theia.lookup.getTableSize(self.group_velocity)
        size += theia.lookup.getTableSize(self.absorption_coef)
        size += theia.lookup.getTableSize(self.scattering_coef)
        size += theia.lookup.getTableSize(self.phase_function)
        size += theia.lookup.getTableSize(self.phase_sampling)
        # check if we need some padding bytes to comply with the
        # 8 byte alignment of the Medium struct
        if size % 8 != 0:
            size += 8 - (size % 8)
        # lastly, add the size of the struct itself
        size += sizeof(Medium.GLSL)
        # done
        return size


class Material:
    """
    Class holding information about the material of a geometry. In general a
    geometry separates space into an "inside" and an "outside". Materials
    assign a Medium to each of them.

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
    """

    class GLSL(Structure):
        """The corresponding structure used in shaders"""

        _fields_ = [
            ("inside", c_uint64),  # buffer reference
            ("outside", c_uint64),  # buffer reference
        ]

    def __init__(
        self,
        name: str,
        inside: Union[Medium, str, None],
        outside: Union[Medium, str, None],
    ) -> None:
        # store properties
        self.name = name
        self.inside = inside
        self.outside = outside

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
    def byte_size(self) -> int:
        """
        Amount of bytes needed to store the material on the GPU.
        This does not include the memory required to store the referenced
        media as they may be shared among multiple materials.
        """
        return sizeof(Material.GLSL)


def serializeMedium(medium: Medium, dst: int, gpu_dst: int) -> tuple[int, int, int]:
    """
    Serializes the given medium and writes the result to the memory at the
    address dst.

    Parameters
    ----------
    medium: Medium
        Medium to be serialized.
    dst: int
        Memory address the result should be written to
    gpu_dst: int
        The corresponding device address on the GPU, i.e. if the memory at
        dst get's uploaded to GPU, gpu_dst points to the first byte.

    Returns
    -------
    p_medium: int
        GPU address pointing to the start of the destination tensor
    new_dst: int
        address one byte after the last written to dst
    new_gpu_dst: int
        equivalent to new_dst on the GPU
    """
    # local copies safe to modify
    p = int(dst)
    p_gpu = int(gpu_dst)
    nbytes_written = 0
    # create new GLSL equivalent. We'll fill it as we serialize the tables
    glsl = Medium.GLSL(lambda_min=medium.lambda_min, lambda_max=medium.lambda_max)

    # little helper to make serializing table more easy
    def processTable(table) -> int:
        nonlocal p, p_gpu, nbytes_written
        # skip if None
        if table is None:
            return 0
        # copy table
        data = theia.lookup.createTable(table)
        memmove(p, data.ctypes.data, data.nbytes)
        # update pointers
        p_table = int(p_gpu)
        p += data.nbytes
        p_gpu += data.nbytes
        nbytes_written += data.nbytes
        # return gpu pointer pointing to this table
        return p_table

    # serialize tables
    glsl.n = processTable(medium.refractive_index)
    glsl.vg = processTable(medium.group_velocity)
    glsl.mu_a = processTable(medium.absorption_coef)
    glsl.mu_s = processTable(medium.scattering_coef)
    glsl.phase = processTable(medium.phase_function)
    glsl.phase_sampling = processTable(medium.phase_sampling)
    # sanity check
    assert p - dst == nbytes_written
    assert p_gpu - gpu_dst == nbytes_written

    # check if we need to add padding to fit Medium's 8 byte alignment
    if nbytes_written % 8 != 0:
        padding = 8 - (nbytes_written % 8)
        p += padding
        p_gpu += padding
        nbytes_written += padding
    # sanity check
    assert p_gpu % 8 == 0

    # copy medium struct
    memmove(p, addressof(glsl), sizeof(Medium.GLSL))
    nbytes_written += sizeof(Medium.GLSL)
    # sanity check
    assert nbytes_written == medium.byte_size

    # Done
    return (p_gpu, p + sizeof(Medium.GLSL), p_gpu + sizeof(Medium.GLSL))


def bakeMaterials(
    *materials: Iterable[Material],
    media: Union[Iterable[Medium], dict[str, Medium]] = [],
) -> tuple[hp.ByteTensor, dict[str, int], dict[str, int]]:
    """
    Processes the given materials and uploads them to the gpu.

    Parameters
    ----------
    materials: Iterable[Material]
        list of materials to be uploaded to the gpu.
    mediums: Union[Iterable[Medium], dict[str, Medium]] = []
        Additional media to be uploaded to the gpu. Media that are referenced
        by name by materials must be contained in here.

    Returns
    -------
    tensor: hephaistos.ByteTensor
        The tensor holding the actual data on the gpu. Addresses of the baked
        materials and media become invalid if this tensor gets deleted or
        dropped.
    materials: dict[str, int]
        Dictionary of the baked materials' addresses in the GPU memory using
        their names as the key.
    media: dict[str, int]
        Dictionary of the baked media's addresses in the GPU memory using
        their names as the key.
    """
    # Start with collecting all media
    if type(media) == list:
        media = {medium.name: medium for medium in media}
    elif type(media) != dict:
        raise RuntimeError("Media parameter has wrong type!")
    for mat in materials:
        if type(mat.inside) == Medium and mat.inside.name not in media:
            media[mat.inside.name] = mat.inside
        if type(mat.outside) == Medium and mat.outside.name not in media:
            media[mat.outside.name] = mat.outside

    # calculate the size of the tensor we'll need
    # start with the sizes needed for the media
    size = sum([m.byte_size for m in media.values()])
    # check if we need padding in between to fit Material's 8 byte alignment
    padding = 0
    if size % 8 != 0:
        padding = 8 - (size % 8)
        size += padding
    # finally add the size of the materials
    size += sum([m.byte_size for m in materials])

    # reserve memory
    staging = hp.RawBuffer(size)
    tensor = hp.ByteTensor(size)
    # fetch addresses
    dst = staging.address
    dst_gpu = tensor.address

    # serialize media
    media_dict: dict[str, int] = {}
    for name, medium in media.items():
        media_dict[name], dst, dst_gpu = serializeMedium(medium, dst, dst_gpu)
    # add padding (if needed)
    dst += padding
    dst_gpu += padding
    # sanity check
    assert dst_gpu % 8 == 0

    # util function for fetching material address
    def getMedAdr(m, err_name):
        # if no medium is specified -> special value 0 (vacuum)
        if m is None:
            return 0
        # get medium's name
        name = None
        if type(m) is Medium:
            name = m.name
        elif type(m) is str:
            name = m
        else:
            raise RuntimeError(f"Unknown type medium for {err_name}")
        # fetch medium from dict
        result = media_dict.get(name)
        if result is None:
            assert type(m) is str  # Otherwise we screwed up...
            raise RuntimeError(f"Unknown material {m} for {err_name}")
        return result

    # serialize materials
    material_dict: dict[str, int] = {}
    for material in materials:
        # create GLSL equivalent
        glsl = Material.GLSL(
            inside=getMedAdr(material.inside, f"{material.name}.inside"),
            outside=getMedAdr(material.outside, f"{material.name}.outside"),
        )
        # copy to buffer
        memmove(dst, addressof(glsl), sizeof(Material.GLSL))
        # safe address in dict
        material_dict[material.name] = dst_gpu
        # update pointers
        dst += sizeof(Material.GLSL)
        dst_gpu += sizeof(Material.GLSL)

    # sanity check
    assert dst - staging.address == size
    assert dst_gpu - tensor.address == size

    # upload data
    hp.execute(hp.updateTensor(staging, tensor))

    # done
    return tensor, material_dict, media_dict


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
        Calculates the group velocity in m/s for the given wavelengths in nm.
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

    def phase_function(self, cos_theta: npt.ArrayLike) -> Union[npt.NDArray, None]:
        """
        Evaluates the phase functions for the given values of cos theta
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
            phase_function=self.phase_function(t),
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
        Calculates the group velocity in m/s for the given wavelengths in nm
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

    def phase_function(self, cos_theta: npt.ArrayLike) -> npt.NDArray:
        """
        Evaluates the phase function for the given angles as cos(theta).
        Normalized with respect to unit sphere.
        """
        return (
            (1.0 - self.g ** 2)
            / np.power(1.0 + self.g ** 2 - 2 * self.g * cos_theta, 1.5)
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
                + self.g ** 2
                - ((1.0 - self.g ** 2) / (1 + self.g - 2.0 * self.g * eta)) ** 2
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

    def phase_function(self, cos_theta: npt.ArrayLike) -> npt.NDArray:
        """Evaluates the phase function for the given angles mu = cos(theta)"""
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
        C = (1 - d_180_nu) * (3 * x ** 2 - 1)
        D = 16 * np.pi * (d_180 - 1) * d_180_nu
        return A / B + C / D

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
            + self.L2 * (L ** 2)
            + self.LM2 / (L ** 2)
            + self.LM4 / (L ** 4)
            + self.LM6 / (L ** 6)
            + self.T1 * T
            + self.T2 * (T ** 2)
            + self.T3 * (T ** 3)
            + self.T4 * (T ** 4)
            + self.TL * T * L
            + self.T2L * (T ** 2) * L
            + self.T3L * (T ** 3) * L
        )
        N2 = (
            self.S0 * S
            + self.S1LM2 * S / (L ** 2)
            + self.S1T * S * T
            + self.S1T2 * S * (T ** 2)
            + self.S1T3 * S * (T ** 3)
            + self.STL * S * T * L
        )
        N3 = (
            self.P1 * p
            + self.P2 * (p ** 2)
            + self.PLM2 * p / (L ** 2)
            + self.PT * p * T
            + self.PT2 * p * (T ** 2)
            + self.P2T2 * (p ** 2) * (T ** 2)
        )
        N4 = self.P1S * p * S + self.PTS * p * T * S + self.PT2S * p * (T ** 2) * S
        return N1 + N2 + N3 + N4

    def group_velocity(self, wavelength: npt.ArrayLike) -> npt.NDArray:
        """Calculates the group velocity"""
        # formula expects wavelengths in micrometers -> convert
        L = wavelength / 1000.0
        T = self.temperature
        p = self.pressure
        S = self.salinity
        # G_i = dN_i/dL
        G1 = (
            2.0 * self.L2 * L
            - 2.0 * self.LM2 / (L ** 3)
            - 4.0 * self.LM4 / (L ** 5)
            - 6.0 * self.LM6 / (L ** 7)
            + self.TL * T
            + self.T2L * (T ** 2)
            + self.T3L * (T ** 3)
        )
        G2 = -2.0 * self.S1LM2 * S / (L ** 3) + self.STL * S * T
        G3 = -2.0 * self.PLM2 * p / (L ** 3)
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
