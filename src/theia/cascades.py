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
class CascadeParameters:
    """Parameterization of electro-magnetic and hadronic showers"""

    # longitudinal emission profile
    a_long: float
    b_long: float

    # scaling factor applied to get the total amount of photons
    energyScale: float = 1.0

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

    # angular emission profile
    # consists of a and b param
    # each fitted a linear regression w.r.t.: p = m * log(E) + l
    a_angluar_shift: float = 0.5375
    a_angular_slope: float = 0.0
    b_angular_shift: float = 3.302
    b_angular_slope: float = 0.0

    # hadronic parameterization
    E0: float = 0.0
    f0: float = 0.0
    m: float = 0.0
    rms0: float = 0.0
    gamma: float = 0.0


def createCascadeParameters(
    p: CascadePrimaryParticle,
    E: float,
    X0: float = X0_water,
    density: float = rho_water,
    uRandom: float | None = None,
) -> CascadeParameters:
    """
    Calculates the cascade parameters for the given energy

    Parameters
    ----------
    p: CascadePrimaryParticle
        The primary particle causing the cascade
    E: float
        The energy of the primary particle
    X0: float
        The radiation length of the medium in which the cascade happens
    uRandom: float | None, default=None
        Random number [0,1) used to sample the electromagnetic fraction in
        hadronic showers. If None, draws a random number on its own. Ignored
        for purely electromagnetic cascades.
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

    # calculate energy scale
    energyScale = 5.321 * 0.91 / density * E

    # sample the hadronic fraction
    if p.E0 != 0.0:
        F = 1.0 - (1.0 - p.f0) * (E / p.E0) ** (-p.m)
        sigmaF = p.rms0 * logE ** (-p.gamma)
        # ensure the hadronic fraction is within [0,1]
        # we do this by scaling uRandom to the corresponding range
        rv = norm(loc=F, scale=sigmaF)
        a, b = rv.cdf(0.0), rv.cdf(1.0)
        if uRandom is None:
            uRandom = np.random.rand()
        uRandom = uRandom * (b - a) + a
        energyScale *= rv.ppf(uRandom)

    return CascadeParameters(a_long, b_long, energyScale, a_angular, b_angular)


# EM Cascades
EMinus: Final[CascadePrimaryParticle] = CascadePrimaryParticle(
    alpha_long=2.01849,
    beta_long=1.45469,
    b_long=0.63207,
    a_angluar_shift=0.53734995,
    a_angular_slope=0.0,
    b_angular_shift=3.30382993,
    b_angular_slope=0.0,
)
"""Parameterization of an electro magnetic cascade initiated by a electron"""
EPlus: Final[CascadePrimaryParticle] = CascadePrimaryParticle(
    alpha_long=2.00035,
    beta_long=1.45501,
    b_long=0.63008,
    a_angluar_shift=0.5367158,
    a_angular_slope=0.0,
    b_angular_shift=3.30484209,
    b_angular_slope=0.0,
)
"""Parameterization of an electro magnetic cascade initiated by a positron"""
Gamma: Final[CascadePrimaryParticle] = CascadePrimaryParticle(
    alpha_long=2.83923,
    beta_long=1.34031,
    b_long=0.645263,
    a_angluar_shift=0.53841841,
    a_angular_slope=0.0,
    b_angular_shift=3.29619817,
    b_angular_slope=0.0,
)
"""Parameterization of an electro magnetic cascade initiated by a gamma photon"""

# Hadronic Cascades
PiPlus: Final[CascadePrimaryParticle] = CascadePrimaryParticle(
    alpha_long=1.58357292,
    beta_long=0.96447937,
    b_long=0.33833116,
    E0=0.18791678,
    f0=0.30974123,
    m=0.16267529,
    rms0=0.95899551,
    gamma=1.35589541,
    a_angluar_shift=1.0299732199972658,
    a_angular_slope=-0.08806219920032332,
    b_angular_shift=3.102713004779744,
    b_angular_slope=-0.12229465620485062,
)
"""Parameterization of a hadronic cascade initiated by a positive charged pion"""
PiMinus: Final[CascadePrimaryParticle] = CascadePrimaryParticle(
    alpha_long=1.69176636,
    beta_long=0.93953506,
    b_long=0.34108075,
    E0=0.19826506,
    f0=0.31859323,
    m=0.16218006,
    rms0=0.94033488,
    gamma=1.35070162,
    a_angluar_shift=1.0412256610000645,
    a_angular_slope=-0.09187703681909758,
    b_angular_shift=3.086039699134421,
    b_angular_slope=-0.11874011144663844,
)
"""Parameterization of a hadronic cascade initiated by a negative charged pion"""
K0_Long: Final[CascadePrimaryParticle] = CascadePrimaryParticle(
    alpha_long=1.95948974,
    beta_long=0.80440041,
    b_long=0.34535151,
    E0=0.21687243,
    f0=0.27724987,
    m=0.16861530,
    rms0=1.00318874,
    gamma=1.37528605,
    a_angluar_shift=1.0591474180300977,
    a_angular_slope=-0.09635256670474648,
    b_angular_shift=3.2258115113151793,
    b_angular_slope=-0.15816716921465757,
)
"""Parameterization of a hadronic cascade initiated by a long lived neutral Kaon"""
PPlus: Final[CascadePrimaryParticle] = CascadePrimaryParticle(
    alpha_long=1.47495778,
    beta_long=0.93140483,
    b_long=0.35226706,
    E0=0.29579368,
    f0=0.02455403,
    m=0.19373018,
    rms0=1.01619344,
    gamma=1.45477346,
    a_angluar_shift=1.1011740152471974,
    a_angular_slope=-0.10085388007414371,
    b_angular_shift=3.284514006559762,
    b_angular_slope=-0.16972413208423504,
)
"""Parameterization of a hadronic cascade initiated by a proton"""
PMinus: Final[CascadePrimaryParticle] = CascadePrimaryParticle(
    alpha_long=1.92249171,
    beta_long=0.77601150,
    b_long=0.34969748,
    E0=0.29579368,
    f0=0.02455403,
    m=0.19373018,
    rms0=1.01094637,
    gamma=1.50438415,
    a_angluar_shift=1.1574216500437113,
    a_angular_slope=-0.11090280215147694,
    b_angular_shift=3.5079727644060794,
    b_angular_slope=-0.22892116764330248,
)
"""Parameterization of a hadronic cascade initiated by a neutron"""
Neutron: Final[CascadePrimaryParticle] = CascadePrimaryParticle(
    alpha_long=1.57739060,
    beta_long=0.93556570,
    b_long=0.35269455,
    E0=0.66725124,
    f0=0.17559033,
    m=0.19263595,
    rms0=1.01414337,
    gamma=1.45086895,
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
    particle: Particle, **kwargs
) -> tuple[Type[LightSource], PipelineParams] | None:
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

    # assemble and return pipeline params
    name = kwargs.get("lightSourceName", "lightSource")
    if name:
        name += "__"
    params = {
        f"{name}startPosition": particle.position,
        f"{name}startTime": particle.startTime,
        f"{name}endPosition": endPos,
        f"{name}endTime": endTime,
        f"{name}muonEnergy": particle.energy,
    }
    return (MuonTrackLightSource, params)


def _createCascadeParams(
    particle: Particle, **kwargs
) -> tuple[Type[LightSource], PipelineParams] | None:
    # fetch cascade parameterization
    primary = getCascadeParamsFromParticleType(particle.particleType)
    if primary is None:
        return None
    # get params
    x0 = kwargs.get("x0", X0_water)
    rho = kwargs.get("density", rho_water)
    cascade_params = createCascadeParameters(primary, particle.energy, x0, rho)
    # normalize direction
    dx, dy, dz = particle.direction
    l = np.sqrt(dx**2 + dy**2 + dz**2)
    direction = (dx / l, dy / l, dz / l)

    # assemble and return pipeline params
    name = kwargs.get("lightSourceName", "lightSource")
    if name:
        name += "__"
    params = {
        f"{name}startPosition": particle.position,
        f"{name}startTime": particle.startTime,
        f"{name}direction": direction,
        f"{name}energyScale": cascade_params.energyScale,
        f"{name}a_angular": cascade_params.a_angular,
        f"{name}b_angular": cascade_params.b_angular,
        f"{name}a_long": cascade_params.a_long,
        f"{name}b_long": cascade_params.b_long,
    }
    return (ParticleCascadeLightSource, params)


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
) -> tuple[Type[LightSource], PipelineParams]:
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

    Returns
    -------
    lightSource: Type[LightSource]
        Light source to which the parameterization applies
    params: PipelineParams
        Parameterization of the light source
    """
    # assemble kwargs
    kwargs = {
        "x0": x0,
        "density": density,
        "lightSourceName": lightSourceName,
    }
    # try every converter until we are successful
    for convert in _converters:
        if (res := convert(particle, **kwargs)) is not None:
            return res
    # tried everything but nothing worked
    raise ValueError(f"Could not create params from particle '{particle}'!")
