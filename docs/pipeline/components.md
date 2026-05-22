---
icon: lucide/puzzle
---

# Components

The simulation in theia is split into multiple components. Each components
contributes a piece to the complete simulation code and exposes an interface
to its configurable state. Only at runtime all components are brought together to
form the complete simulation. This makes it easier to both add new components as
well as to reuse common code for different simulation configurations as they
require different but likely overlapping subsets of components.

The following gives a short outline about the type of components present in
`theia`.

## Tracer

At the core of a simulation pipeline is usually a `Tracer`. Its sole
responsibility is to trace light paths delegating other tasks such as
sampling the light source to other components of the corresponding type.
Once a suitable light path has been sampled it generates a hit and passes it to
the specified [HitResponse](components.md#hit-response).

- `SceneBackwardTracer`
- `SceneForwardTracer`
- `SceneBackwardTargetTracer`
- `SceneDirectTracer`
- `VolumeBackwardTracer`
- `VolumeForwardTracer`
- `VolumeDirectTracer`

## Ray Model

Ray models specify the physical quantities the tracer keeps track of during
path sampling and the ones reported in hits, such as expected photon count and
propagation time. It may only support forward or backward tracing or both.

- `UnpolarizedRay`

## Light Source

Light sources model the emission of light as a function of time including the
light ray's direction and starting time. In forward mode the light source is
free to choose any direction, whereas in backward mode it must be in the
direction of a specified _observer_ point. It is free to only implement either
of these mode or both. This is indicated by whether the corresponding
`backwardSourceCode` and `forwardSourceCode` properties return `None` or not.

- `CherenkovLightSource`
- `MuonTrackLightSource`
- `ParticleCascadeLightSource`
- `PencilLightSource`
- `SphericalLightSource`
- `StreamingHostLightSource`

## Camera

Cameras are used in backward and direct light tracing to sample the hit position
at the detector. In the former this includes the initial ray direction used for
path tracing. For direct light tracing this is a two step process: First the
camera samples the hit point without direction and then takes the direction of
the incident sampled from the light source to generate a detector hit. Note that
the support for direct lighting is optional and is reported via the
`supportDirect` property.

!!! tip
    The position given to the tracer does not need to agree with the hit
    position. This allows to include optical effects such as lenses or glass
    housing into the camera.

- `ConeCamera`
- `DiskCamera`
- `FlatCamera`
- `HostCamera`
- `MeshCamera`
- `PencilCamera`
- `PointCamera`
- `SphereCamera`

## Wavelength Source

A wavelength source is used to sample a wavelength for each light path. Its
separation from the light source allows it include other effects such as the
wavelength response from the detector. A common method is to combine the
emission spectrum of the light source with the response from the detector
making it possible to importance sample both.

- `ConstWavelengthSource`
- `FunctionWavelengthSource`
- `HostWavelengthSource`
- `StreamingHostWavelengthSource`
- `UniformWavelengthSource`

## Hit Response

Hit responses are the final component in tracers. They take the hits produced by
the tracer and model an appropriate detector response. Since this is the final
step hit responses are free in what they do with the hit. For instance, they can
simply save the hit for later processing or accumulate them into a histogram.

- `HistogramHitResponse`
- `HitRecorder`
- `IntegratingHitResponse`
- `KernelHistogramHitResponse`
- `StoreTimeHitResponse`
- `StoreValueHitResponse`

## Target

Targets are used in volume tracing to substitute the missing scene and its
geometries. It provides methods to the tracer for intersecting rays with the
target, sampling directions towards and to determine whether a start point of
the tracer is occluded by the target and should be discarded.

- `DiskTarget`
- `FlatTarget`
- `InnerSphereTarget`
- `SphereTarget`

## Target Guide

In forward tracing as a method to increase performance at each volume scatter
event an alternative light path connecting this event to detector is sampled.
Because target geometries in scenes may become very complex directly sampling
them for hit points might becomes unfavorable. Instead target guides are used
as proxy to determine promising direction from a giving position. The actual
hit point is then determined by intersecting the ray with the scene using
ray tracing hardware.

- `DiskTargetGuide`
- `FlatTargetGuide`
- `SphereTargetGuide`

## Random Number Generator

Random number generators are responsible for creating the random numbers
necessary in Monte Carlo simulations. They of course are purely deterministic
only appearing to be random and will always return the same number for the same
parameters. Within a single batch of the simulation each light path has its own
stream of random numbers advanced by a common index. In order for the next batch
to get fresh numbers this index must be advanced through the `offset` parameter.
This can be done automatically by the RNG via setting the `autoAdvance`
parameter accordingly. Usually one uses here the tracer's `nRNGSamples` property
that tells how many random numbers per stream and batch are used by the tracer.

- `PhiloxRNG`
- `SobolQRNG`
