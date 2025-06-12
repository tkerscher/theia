from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import theia.units as u
from theia.light import LightSource, MuonTrackLightSource, ParticleCascadeLightSource

from scipy.stats import norm

from hephaistos.pipeline import PipelineParams
from typing import Final, Type

"""
Parameterization shown here are based on L. Raedel's master thesis and
subsequent papers [1-3]. See also noteboooks/track_angular_dist_fit.ipynb

[1] L. Raedel "Simulation Studies of the Cherenkov Light Yield from Relativistic
    Particles in High-Energy Neutrino Telescopes with Geant 4" (2012)
[2] L. Raedel, C. Wiebusch: "Calculation of the Cherenkov light yield from low
    energetic secondary particles accompanying high-energy muons in ice and
    water with Geant4 simulations" (2012) arXiv:1206.5530v2
[3] L. Raedel, C. Wiebusch: "Calculation of the Cherenkov light yield from
    electromagnetic cascades in ice with Geant4" (2013) arXiv:1210.5140v2
"""

__all__ = [
    "CascadeLightYield",
    "CascadeParameters",
    "CascadePrimaryParticle",
    "EMinus",
    "EPlus",
    "Gamma",
    "K0_Long",
    "Neutron",
    "Particle",
    "ParticleType",
    "PiMinus",
    "PiPlus",
    "PMinus",
    "PPlus",
    "X0_ice",
    "X0_water",
    "createCascadeParameters",
    "createParamsFromParticle",
    "getCascadeParamsFromParticleType",
    "rho_ice",
    "rho_water",
]


def __dir__():
    return __all__


# Radiation lengths
X0_ice: Final[float] = 39.75 * u.cm
"""Radiation length in ice"""
X0_water: Final[float] = 36.08 * u.cm
"""Radiation length in water"""

# medium densities [g/cm3]
rho_ice: Final[float] = 0.91
"""Density of ice"""
rho_water: Final[float] = 1.039  # deep sea water
"""Density of sea water"""


class ParticleType(IntEnum):
    """Particle IDs following the PDG Monte Carlo particle numbering scheme"""

    UNKNOWN = 0
    GAMMA = 22
    E_PLUS = -11
    E_MINUS = 11
    MU_PLUS = -13
    MU_MINUS = 13
    TAU_PLUS = -15
    TAU_MINUS = 15
    PI_0 = 111
    PI_PLUS = 211
    PI_MINUS = -211
    K0_LONG = 130
    NEUTRON = 2112
    P_PLUS = 2212
    P_MINUS = -2212


@dataclass
class Particle:
    """Structure describing a particle"""

    particleType: ParticleType
    """Type of the particle"""

    position: tuple[float, float, float]
    """Position of the particle"""
    direction: tuple[float, float, float]
    """Direction the particle travels"""

    energy: float
    """Energy of the particle"""
    startTime: float = 0.0
    """Time point this particle starts to exists"""
    length: float = float("NaN")
    """Track length if applicable or `NaN` otherwise"""
    speed: float = 1.0 * u.c


@dataclass
class CascadeLightYield:
    """Structure describing the light yield from a cascade"""

    effectiveLength: float
    """Length of equivalent Cherenkov track"""

    effectiveLengthStd: float = 0.0
    """Fluctuation in the effective length described by standard deviation"""


@dataclass
class CascadeParameters:
    """Parameterization of electromagnetic and hadronic showers"""

    # longitudinal emission profile
    a_long: float
    b_long: float

    # effective length, i.e. length of equivalent Cherenkov track producing the
    # same amount of photons. Modeled to be Gaussian distributed
    effectiveLength: float
    effectiveLengthStd: float = 0.0

    # angular emission profile
    # default is average of EM cascades
    a_angular: float = 0.5375
    b_angular: float = 3.302


@dataclass(frozen=True)
class CascadePrimaryParticle:
    """Properties of primary particle used to calculate cascade parameters"""

    # longitudinal emission profile
    alpha_long: float
    beta_long: float
    b_long: float

    # effective length with default for EM
    alpha_length: float = 5.321
    beta_length: float = 1.0
    alpha_length_std: float = 5.727e-2
    beta_length_std: float = 0.5

    # angular emission profile
    # consists of a and b param
    # each fitted a linear regression w.r.t.: p = m * log(E) + l
    a_angluar_shift: float = 0.5375
    a_angular_slope: float = 0.0
    b_angular_shift: float = 3.302
    b_angular_slope: float = 0.0


def createCascadeParameters(
    p: CascadePrimaryParticle,
    E: float,
    X0: float = X0_water,
    density: float = rho_water,
) -> CascadeParameters:
    """
    Calculates the cascade parameters for the given particle and energy.

    Parameters
    ----------
    p: CascadePrimaryParticle
        The primary particle causing the cascade
    E: float
        The energy of the primary particle
    X0: float
        The radiation length of the medium in which the cascade happens
    """
    logE = max(0.0, np.log10(E))

    # longitudinal parameters
    a_long = p.alpha_long + p.beta_long * logE
    b_long = X0 / p.b_long

    # angular parameters
    # see notebooks/track_angular_dist_fit.ipynb
    # or shader/lightsource.particles.common.glsl
    a_angular = p.a_angluar_shift * logE + p.a_angluar_shift
    b_angular = p.b_angular_slope * logE + p.a_angluar_shift

    # calculate effective length
    rhoScale = 0.91 / density  # See eq (9) in [3]
    effectiveLength = p.alpha_length * rhoScale * (E**p.beta_length)
    effectiveLengthStd = p.alpha_length_std * rhoScale * (E**p.beta_length_std)

    return CascadeParameters(
        a_long,
        b_long,
        effectiveLength,
        effectiveLengthStd,
        a_angular,
        b_angular,
    )


# EM Cascades
EMinus: Final[CascadePrimaryParticle]
"""Parameterization of an electromagnetic cascade initiated by an electron"""
EMinus = CascadePrimaryParticle(
    alpha_long=2.01849,
    beta_long=1.45469,
    b_long=0.63207,
    alpha_length=5.3207078881,
    beta_length=1.00000211,
    alpha_length_std=0.0578170887,
    beta_length_std=0.5,
    a_angluar_shift=0.53734995,
    a_angular_slope=0.0,
    b_angular_shift=3.30382993,
    b_angular_slope=0.0,
)
EPlus: Final[CascadePrimaryParticle]
"""Parameterization of an electro magnetic cascade initiated by a positron"""
EPlus = CascadePrimaryParticle(
    alpha_long=2.00035,
    beta_long=1.45501,
    b_long=0.63008,
    alpha_length=5.3211320598,
    beta_length=0.99999254,
    alpha_length_std=0.0573419669,
    beta_length_std=0.5,
    a_angluar_shift=0.5367158,
    a_angular_slope=0.0,
    b_angular_shift=3.30484209,
    b_angular_slope=0.0,
)
Gamma: Final[CascadePrimaryParticle]
"""Parameterization of an electro magnetic cascade initiated by a gamma photon"""
Gamma = CascadePrimaryParticle(
    alpha_long=2.83923,
    beta_long=1.45501,
    b_long=0.64526,
    alpha_length=5.3208540905,
    beta_length=0.99999877,
    alpha_length_std=0.0566586567,
    beta_length_std=0.5,
    a_angluar_shift=0.53841841,
    a_angular_slope=0.0,
    b_angular_shift=3.29619817,
    b_angular_slope=0.0,
)

# Hadronic cascades
PiPlus: Final[CascadePrimaryParticle]
"""Parameterization of a hadronic cascade initiated by a positive charged pion"""
PiPlus = CascadePrimaryParticle(
    alpha_long=1.58357292,
    beta_long=0.96447937,
    b_long=0.33833116,
    alpha_length=3.3355182722,
    beta_length=1.03662217,
    alpha_length_std=1.1920455395,
    beta_length_std=0.80772057,
    a_angluar_shift=1.0299732199972658,
    a_angular_slope=-0.08806219920032332,
    b_angular_shift=3.102713004779744,
    b_angular_slope=-0.12229465620485062,
)
PiMinus: Final[CascadePrimaryParticle]
"""Parameterization of a hadronic cascade initiated by a negative charged pion"""
PiMinus = CascadePrimaryParticle(
    alpha_long=1.69176636,
    beta_long=0.93953506,
    b_long=0.34108075,
    alpha_length=3.3584489578,
    beta_length=1.03584394,
    alpha_length_std=1.2250188073,
    beta_length_std=0.80322520,
    a_angluar_shift=1.0412256610000645,
    a_angular_slope=-0.09187703681909758,
    b_angular_shift=3.086039699134421,
    b_angular_slope=-0.11874011144663844,
)
K0_Long: Final[CascadePrimaryParticle]
"""Parameterization of a hadronic cascade initiated by a long lived neutral Kaon"""
K0_Long = CascadePrimaryParticle(
    alpha_long=1.95948974,
    beta_long=0.80440041,
    b_long=0.34535151,
    alpha_length=3.2600450524,
    beta_length=1.03931457,
    alpha_length_std=1.2141970572,
    beta_length_std=0.80779629,
    a_angluar_shift=1.0591474180300977,
    a_angular_slope=-0.09635256670474648,
    b_angular_shift=3.2258115113151793,
    b_angular_slope=-0.15816716921465757,
)
PPlus: Final[CascadePrimaryParticle]
"""Parameterization of a hadronic cascade initiated by a proton"""
PPlus = CascadePrimaryParticle(
    alpha_long=1.92249171,
    beta_long=0.77601150,
    b_long=0.34969748,
    alpha_length=2.8737183922,
    beta_length=1.05172118,
    alpha_length_std=0.8804581378,
    beta_length_std=0.82445572,
    a_angluar_shift=1.1574216500437113,
    a_angular_slope=-0.11090280215147694,
    b_angular_shift=3.5079727644060794,
    b_angular_slope=-0.22892116764330248,
)
PMinus: Final[CascadePrimaryParticle]
"""Parameterization of a hadronic cascade initiated by a neutron"""
PMinus = CascadePrimaryParticle(
    alpha_long=1.92249171,
    beta_long=0.77601150,
    b_long=0.34969748,
    alpha_length=3.0333074914,
    beta_length=1.04322206,
    alpha_length_std=1.1323088104,
    beta_length_std=0.77134060,
    a_angluar_shift=1.1574216500437113,
    a_angular_slope=-0.11090280215147694,
    b_angular_shift=3.5079727644060794,
    b_angular_slope=-0.22892116764330248,
)
Neutron: Final[CascadePrimaryParticle]
"""Parameterization of a hadronic cascade initiated by a neutron"""
Neutron = CascadePrimaryParticle(
    alpha_long=1.57739060,
    beta_long=0.93556570,
    b_long=0.35269455,
    alpha_length=2.7843854660,
    beta_length=1.05582906,
    alpha_length_std=0.9322787137,
    beta_length_std=0.81776503,
    a_angluar_shift=1.1292267334081203,
    a_angular_slope=-0.10876633838986713,
    b_angular_shift=3.4157386880981093,
    b_angular_slope=-0.20638832466150736,
)

_cascadeParticlesMap = {
    ParticleType.GAMMA: Gamma,
    ParticleType.E_MINUS: EMinus,
    ParticleType.E_PLUS: EPlus,
    ParticleType.PI_0: Gamma,  # decays immediately to two gammas
    ParticleType.PI_PLUS: PiPlus,
    ParticleType.PI_MINUS: PiMinus,
    ParticleType.K0_LONG: K0_Long,
    ParticleType.P_PLUS: PPlus,
    ParticleType.P_MINUS: PMinus,
    ParticleType.NEUTRON: Neutron,
}


def getCascadeParamsFromParticleType(t: ParticleType) -> CascadePrimaryParticle | None:
    """
    Returns the cascade parameterization for the given particle type or `None`
    if no known parameterization exist.
    """
    return _cascadeParticlesMap.get(t)


# particle to params converters
_trackParticles = {
    ParticleType.MU_PLUS,
    ParticleType.MU_MINUS,
    ParticleType.TAU_PLUS,
    ParticleType.TAU_MINUS,
}


def _createTrackParams(
    particle: Particle,
    *,
    name: str = "lightSource",
    uRand: float | None = None,
    **kwargs,
) -> tuple[Type[LightSource], PipelineParams, CascadeLightYield] | None:
    if particle.particleType not in _trackParticles:
        return None

    # calculate end point
    if not particle.length > 0.0:  # This also catches NaN
        raise ValueError("particle is muon like, but no track length was specified!")
    x, y, z = particle.position
    dx, dy, dz = particle.direction
    l = particle.length / np.sqrt(dx**2 + dy**2 + dz**2)  # normalize direction
    endPos = (x + l * dx, y + l * dy, z + l * dz)
    endTime = particle.startTime + particle.length / particle.speed
    # calculate light yield
    scale = 1.1880 + 0.0206 * np.log(particle.energy)
    length = particle.length * scale

    # TODO: Really unsure about this one. Need to revisit...
    #       CLSim uses sqrt of total number of photons, but a) we do not know
    #       that yet here (missing frank tamm) and b) the variance should lie
    #       in the amount of secondary particles produced, not in the amount of
    #       photons (that's constant per particle in a classical view)
    # calculate std of effective length. See eq. 7 in [2]
    std = np.sqrt(particle.length * 0.1 * u.m) * scale  # the 10cm look really weird...

    # sample effective length
    if uRand is not None:
        length += norm.ppf(uRand).item() * std
        # we are always guaranteed to at least get the photons from the bare muon
        # TODO: This is technically no longer gaussian -> check if close enough
        length = max(length, particle.length)
        std = 0.0  # because we sampled a specific one

    # assemble and return pipeline params
    if name:
        name += "__"
    params = {
        f"{name}startPosition": particle.position,
        f"{name}startTime": particle.startTime,
        f"{name}endPosition": endPos,
        f"{name}endTime": endTime,
        f"{name}muonEnergy": particle.energy,
    }
    return (MuonTrackLightSource, params, CascadeLightYield(length, std))


def _createCascadeParams(
    particle: Particle,
    *,
    name: str = "lightSource",
    X0: float = X0_water,
    rho: float = rho_water,
    uRand: float | None = None,
    **kwargs,
) -> tuple[Type[LightSource], PipelineParams, CascadeLightYield] | None:
    # fetch cascade parameterization
    primary = getCascadeParamsFromParticleType(particle.particleType)
    if primary is None:
        return None
    # get params
    cascade_params = createCascadeParameters(primary, particle.energy, X0, rho)
    # sample effective length if needed
    effectiveLength = cascade_params.effectiveLength
    effectiveStd = cascade_params.effectiveLengthStd
    if uRand is not None:
        effectiveLength += norm.ppf(uRand).item() * effectiveStd
        effectiveLength = max(0.0, effectiveLength)  # just to be sure
        effectiveStd = 0.0  # because we sampled a specific one
    lightYield = CascadeLightYield(effectiveLength, effectiveStd / effectiveLength)
    # normalize direction
    dx, dy, dz = particle.direction
    l = np.sqrt(dx**2 + dy**2 + dz**2)
    direction = (dx / l, dy / l, dz / l)

    # assemble and return pipeline params
    if name:
        name += "__"
    params = {
        f"{name}startPosition": particle.position,
        f"{name}startTime": particle.startTime,
        f"{name}direction": direction,
        f"{name}effectiveLength": effectiveLength,
        f"{name}a_angular": cascade_params.a_angular,
        f"{name}b_angular": cascade_params.b_angular,
        f"{name}a_long": cascade_params.a_long,
        f"{name}b_long": cascade_params.b_long,
    }
    return (ParticleCascadeLightSource, params, lightYield)


_converters = [
    _createTrackParams,
    _createCascadeParams,
]


def createParamsFromParticle(
    particle: Particle,
    *,
    x0: float = X0_water,
    density: float = rho_water,
    lightSourceName: str = "lightSource",
    uRand: float | None = None,
) -> tuple[Type[LightSource], PipelineParams, CascadeLightYield]:
    """
    Returns the parameterization and corresponding light source for a given
    particle.

    Parameters
    ----------
    particle: Particle
        Particle to process
    x0: float, default=X0_water
        Radiation length of the surrounding media
    density: float, default=rho_water
        Density of the surrounding media
    lightSourceName: str, default="lightSource"
        Name of the light source as noted in the corresponding pipeline.
        If not empty, parameter names will be prepended with
        '{lightSourceName}__' as expected by a pipeline.
    uRand: float | None, default=None
        Random number used to sample effective length from the underlying
        distribution. If `None`, uses the mean.

    Returns
    -------
    lightSource: Type[LightSource]
        Light source to which the parameterization applies
    params: PipelineParams
        Parameterization of the light source
    yield: CascadeLightYield
        Light yield from the particle
    """
    # assemble kwargs
    kwargs = {
        "x0": x0,
        "density": density,
        "name": lightSourceName,
        "uRand": uRand,
    }
    # try every converter until we are successful
    for convert in _converters:
        if (res := convert(particle, **kwargs)) is not None:
            return res
    # tried everything but nothing worked
    raise ValueError(f"Could not create params from particle '{particle}'!")
