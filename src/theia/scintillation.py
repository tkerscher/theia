from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ctypes import Structure, c_float, c_uint32
from hephaistos.glsl import vec3
from theia.compiler import createPreamble, loadShader

from theia.light import LightSource, WavelengthSource
from theia.material import Medium, MaterialStore
import theia.units as u

__all__ = [
    "EnergyDepositionEvent",
    "ScintillationDepositionEventSource",
    "createTasksFromEvents",
]


def __dir__():
    return __all__


@dataclass
class EnergyDepositionEvent:
    """
    Data class to store energy deposition events in a form that is compatible
    with the output of remage simulations.
    """
    eventId: int
    medium: Medium
    startPosition: tuple[float, float, float]  = (0.0, 0.0, 0.0)
    endPosition: tuple[float, float, float]  = (0.0, 0.0, 0.0)
    startTime: float = 0.0
    startVelocity: float = 0.0
    endVelocity: float = 0.0
    energyDeposition: float = 1.0 * u.keV
    particleCode: int = 11


def createTasksFromEvents(events: list[EnergyDepositionEvent],
                          matStore: MaterialStore,
                          *,
                          doImportanceSampling: bool = False,
                          Repetitions: int = 1,
                          batchesPerPhoton: float = 0.01) -> list[dict]:
    """
    Utility function to prepare the simulation of scintillation light. Takes 
    a list of events and returns a list of tasks which properly configures 
    the light source and wavelength source for a scheduler, assuming a
    `ScintillationDepositionEventSource` and a `SpectrumWavelengthSource` is used.
    Scintillation spectrums, time constants, branching ratios and yields must be
    defined for all scintillating media through their medium properties (see
    `material.OpticalProperties` for the property names

    Parameters
    ----------
    events: list[EnergyDepositionEvent]
        List of events to generate the tasks for.
    matStore: MaterialStore
        Material Store that is used for the simulation. 
    doImportanceSampling: bool = False
        If `False`, each scintillation component of each event will get the
        same number of batches, as specified by Repetitions. If `True`, the number
        of batches for a given component will be propertional to the number of
        photons, as specified by batchesPerPhoton, with each component getting
        at least one batch.
    Repititions: int = 1
        Number of batches to be simulated for each scintillation component of each
        event. Will be ignored of `doImportanceSapling` is `True`.
    batchesPerPhoton: float
        Number of batches sampled per physical photon. At least one batch will be
        generated for each scintillation component of each event. Will be ignored 
        if `doImportanceSampling` is `False`.
    """
    tasks = []
    for event in events:
        # get photon yield and branching ratio based on particle type
        if(event.particleCode == 11):
            branchingRatio =  event.medium.properties["scintillation_branching_ratio_electron"].value[1]
            photonYield = event.medium.properties["scintillation_yield_electron"].value[1]
        elif(event.particleCode == 2212):
            branchingRatio =  event.medium.properties["scintillation_branching_ratio_proton"].value[1]
            photonYield = event.medium.properties["scintillation_yield_proton"].value[1]
        elif(event.particleCode == 1000020040):
            branchingRatio =  event.medium.properties["scintillation_branching_ratio_alpha"].value[1] 
            photonYield = event.medium.properties["scintillation_yield_alpha"].value[1]
        elif(event.particleCode > 1000000000):
            branchingRatio =  event.medium.properties["scintillation_branching_ratio_ion"].value[1] 
            photonYield = event.medium.properties["scintillation_yield_ion"].value[1]
        else:
            raise ValueError(f"Unknown paticle code {event.particleCode}")
        
        if doImportanceSampling:
            num_batches = np.round(branchingRatio * photonYield * event.energyDeposition * batchesPerPhoton)
            Repetitions = np.max([1,num_batches])
        
        # append task
        tasks.append({"lightSource__mediumIdx": matStore.media[event.medium.name],
                      "lightSource__startPosition": event.startPosition,
                      "lightSource__endPosition": event.endPosition,
                      "lightSource__startTime": event.startTime,
                      "lightSource__startVelocity": event.startVelocity,
                      "lightSource__endVelocity": event.endVelocity,
                      "lightSource__energyDeposition": event.energyDeposition,
                      "lightSource__photonYield": photonYield * branchingRatio / Repetitions,
                      "lightSource__timeConstant": event.medium.properties["scintillation_time_constant_1"].value[1],
                      "photons__spectrumTableAddress": event.medium.properties["scintillation_spectrum_1_sampler"].data
                      })
        
        # repeat the same task (Repetitions - 1) times
        if Repetitions > 1:
            tasks += [{}] * (Repetitions - 1)
        
        # check if the second scintillation component exists
        if branchingRatio < 1.0:

            if doImportanceSampling:
                num_batches = np.round(branchingRatio * (1 - photonYield) * event.energyDeposition * batchesPerPhoton)
                Repetitions = np.max([1,num_batches])

            tasks.append({"lightSource__mediumIdx": matStore.media[event.medium.name],
                          "lightSource__startPosition": event.startPosition,
                          "lightSource__endPosition": event.endPosition,
                          "lightSource__startTime": event.startTime,
                          "lightSource__startVelocity": event.startVelocity,
                          "lightSource__endVelocity": event.endVelocity,
                          "lightSource__energyDeposition": event.energyDeposition,
                          "lightSource__photonYield": photonYield * (1 - branchingRatio) / Repetitions,
                          "lightSource__timeConstant": event.medium.properties["scintillation_time_constant_2"].value[1],
                          "photons__spectrumTableAddress": event.medium.properties["scintillation_spectrum_2_sampler"].data
                          })
            
            if Repetitions > 1:
                tasks += [{}] * (Repetitions - 1)

    return tasks


class ScintillationDepositionEventSource(LightSource):
    """
    Samples scintillation light from an energy deposition event. It is assumed 
    that the energy is deposited uniformly along the specified track segment.

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from. Required for forward mode.
    mediumIdx: int
        Index of the medium the light source is located in
    startPosition: (float, float, float), default=(0.0, 0.0, 0.0)
        Position of the start of the track segment.
    endPosition: (float, float, float), default=(0.0, 0.0, 0.0)
        Position of the end of the track segment.
    startTime: float, default=0.0
        Time at the start of the track segment.
    startVelocity: float, default=0.0
        Velocity of the particle at the start of the track segment.
    endVelocity: float, default=0.0
        Velocity of the particle at the end of the track segment.
    energyDeposition: float, default=1.0 keV
        Amount of energy deposited in the event.
    photonYield: float, default=10000/MeV
        Number of generated photons per deposited energy.
    timeConstant: float, default=0.0
        Time constant of the exponential decay.
    usePhotonCount: bool, default=True
        If `True` sampled radiance has units of number photons, otherwise eV
    emitParticles: bool, default=False
        Whether to emit particles instead of rays.
    """

    class LightParams(Structure):
        _fields_ = [
            ("startPosition", vec3),
            ("endPosition", vec3),
            ("mediumIdx", c_uint32),
            ("startTime", c_float),
            ("startVelocity", c_float),
            ("endVelocity", c_float),
            ("energyDeposition", c_float),
            ("photonYield", c_float),
            ("timeConstant", c_float),
        ]

    def __init__(
        self,
        wavelengthSource: WavelengthSource | None = None,
        *,
        mediumIdx: int,
        startPosition: tuple[float, float, float] = (0.0, 0.0, 0.0),
        endPosition: tuple[float, float, float] = (0.0, 0.0, 0.0),
        startTime: float = 0.0,
        startVelocity: float = 0.0,
        endVelocity: float = 0.0,
        energyDeposition: float = 1.0 * u.keV,
        photonYield: float = 10 / u.keV,
        timeConstant: float = 0.0,
        usePhotonCount: bool = True,
        emitParticles: bool = False,
    ) -> None:
        lam = wavelengthSource
        super().__init__(
            nRNGForward=0 if lam is None else lam.nRNGSamples + 4,
            nRNGBackward=2,
            wavelengthSource=wavelengthSource,
            params={"LightParams": ScintillationDepositionEventSource.LightParams},
        )
        # check params
        if emitParticles and not usePhotonCount:
            raise ValueError("Must use photon count when emitting particles")

        # save params
        self._usePhotonCount = usePhotonCount
        self._emitParticles = emitParticles

        self.setParams(
            startPosition=startPosition,
            endPosition=endPosition,
            mediumIdx=mediumIdx,
            startTime=startTime,
            startVelocity=startVelocity,
            endVelocity=endVelocity,
            energyDeposition=energyDeposition,
            photon_yield=photonYield,
            timeConstant=timeConstant,
        )

    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def usePhotonCount(self) -> bool:
        """Whether to sample radiance in eV or #photons"""
        return self._usePhotonCount

    @property
    def forwardSourceCode(self) -> str | None:
        if self.wavelengthSource is None:
            return None
        preamble = createPreamble(
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
            SCINTILLATION_USE_ENERGY=not self.usePhotonCount,
        )
        code = loadShader("lightsource/scintillation/deposition_event/forward.glsl")
        return preamble + code
    
    @property
    def backwardSourceCode(self) -> str | None:
        if self.emitsParticles:
            return None
        preamble = createPreamble(
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
            SCINTILLATION_USE_ENERGY=not self.usePhotonCount,
        )
        code = loadShader("lightsource/scintillation/deposition_event/backward.glsl")
        return preamble + code
