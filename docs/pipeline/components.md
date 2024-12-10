# Simulation Components

The Monte Carlo simulation is organized into several components allowing to
mix, match and extend them as necessary. Each component has a type denoted by
their base class that defines the interface they implement. Since they are
meant to be used with [Pipelines](pipeline.md) they all inherit from
[`PipelineStage`](pipeline.md#pipeline-stage). The following introduces each
type used in simulation.

## Tracer

At the core of a simulation pipeline is usually a `Tracer`. Its sole
responsibility is to trace light paths delegating other tasks such as
sampling the light source to other components of the corresponding type.
Once a suitable light path has been sampled it generates a hit and passes it to
the specified [HitResponse](components.md#hit-response).

There are various different tracer available each aiming at different tasks.
But they also have some things in common such as requiring a hit response as
mentioned before. One can generally specify the batch size, that is the number
of light paths tracer per pipeline run. According to that the tracer will
calculate the normalization factor needed to get the correct estimate as well
as a upper limit for the amount of hits generated. The latter is useful if one
wants to save them.

!!! tip
    All tracers have a parameter `blockSize` specifying the number of threads
    in a single GPU workgroup. Adjusting this can increase performance as the
    optimal value is specific to the hardware. It should however be a
    multiple of 32 for most of them.

### Scene vs. Volume Tracer

A big difference in tracing is whether one uses a [scene](../scene.md). While
it allows for multiple media and accurate detector models including reflection
and transmission, it also requires ray tracing capable hardware and is generally
computationally more demanding. Alternatively, one can trace a simple homogenous
medium. In that case the tracer requires a suitable component that determines
whether a ray hit the detector.

The distinction between scene and volume tracer can either happen by making the
scene an optional parameter like in [direct light tracer](components.md#direct-light-tracer)
or, if the implementation differs to much, by different tracers distinguished by
name like for `SceneForwardTracer` and `VolumeForwardTracer`.

### Forward Tracer

Forward tracing is perhaps what most people think of when talking about photon
tracing. A light ray is sampled from a [light source](components.md#light-source)
and traced through a [scene](../scene.md) or volume until it hits a target.
The former case is implemented by `SceneForwardTracer` and the latter by
`VolumeForwardTracer`. For volumes the target is defined by a
[Target](components.md#target) component, whereas for scenes it is defined by a
geometry with material that has the `DETECTOR` bit set. If a scene defines
multiple targets, the active one can be selected by specifying the `targetIdx`
which must match the corresponding `detectorId` to generate hits.

To increase performance of the simulation both tracers are capable of creating
alternative light paths by deliberately connecting scattering events with the
target. As explained with [path integrals](../model.md#path-integrals), since
these are paths of varying length they contribute to different integrals of
the underlying estimator and are thus safe to use without introducing
auto-correlation. Targets support this out of the box. For scenes one must
additionally provide a [target guide](components.md#target-guide).

### Backward Tracer

Backward tracer do not get their initial ray from the light source but sample a
[camera](components.md#camera) as proxy for the detector instead. This has
actually a key benefit: Since we can now define where the light will hit,
methods like importance sampling the detector becomes possible. They can also
show better performance in shadowed detectors, where forward tracer will mostly
produce blocked light paths while here the ray has time to leave the shadow.
Note that light sources may not support backward tracing.

Like it is the case with forward tracing there are two implementations for
backward tracing depending on whether one wants to use a scene:
`SceneBackwardTracer` and `VolumeBackwardTracer`. For the latter a
[target](components.md#target) becomes optional but can be used to simulate
self-shadowing.

!!! warning
    Backward tracing cannot connect specular transmission and reflection to the
    light source. Such paths are missing from the Monte Carlo estimate causing
    bias. It depends on the simulated case basis whether this is acceptable.
    Optionally, the tracer can include direct lighting contributions if
    supported by both the light source and camera.

### Direct Light Tracer

As the name suggest, this tracer estimates the incident light at the detector
that did not undergo any scattering. It does so by first sampling a hit position
from a [camera](components.md#camera) and then the direction of the incident
light from a [light source](components.md#light-source). Both must support this
mode. For incorporating shadowing effects optionally a scene can be specified.

The idea behind this tracer is twofold: It is a very fast estimator and as
direct light often provides the most energy might already be enough. Combined
with other tracer it can also be used to reduce the bias caused by paths these
are missing such as is the case with backward tracing.

### Bidirectional Tracer

Bidirectional path tracing creates two subpaths starting at the light source and
a camera. These subpaths are then combined at their respective volume scattering
events if not occluded by the scene. Its strength lies in scenarios where most
of the light is shadowed or the both the light source and camera is very
directional. It suffers however from slower convergence and should thus only be
used when other tracers perform poorly.

!!! warning
    Since this tracer needs at least two volume scattering events to connect
    the subpaths it will not sample direct light and paths with only one
    scattering. Additionally, like backward tracing it cannot create paths
    that connect specular transmission to reflection to either light source or
    detector.

## Light Source

Light sources model the emission of light as a function of time including the
light ray's direction and starting time. In forward mode the light source is
free to choose any direction, whereas in backward mode it must be in the
direction of a specified _observer_ point. It is free to only implement either
of these mode or both. This is indicated by its `supportForward` and
`supportBackward` properties.

## Wavelength Source

A wavelength source is used to sample a wavelength for each light path. Its
separation from the light source allows it include other effects such as the
wavelength response from the detector. A common method is to combine the
emission spectrum of the light source with the response from the detector
making it possible to importance sample both.

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

## Target

Targets are used in volume tracing to substitute the missing scene and its
geometries. It provides methods to the tracer for intersecting rays with the
target, sampling directions towards and to determine whether a start point of
the tracer is occluded by the target and should be discarded.

## Target Guide

In forward tracing as a method to increase performance at each volume scatter
event an alternative light path connecting this event to detector is sampled.
Because target geometries in scenes may become very complex directly sampling
them for hit points might becomes unfavorable. Instead target guides are used
as proxy to determine promising direction from a giving position. The actual
hit point is then determined by intersecting the ray with the scene using
ray tracing hardware.

## Hit Response

Hit responses are the final component in tracers. They take the hits produced by
the tracer and model an appropriate detector response. Since this is the final
step hit responses are free in what they do with the hit. For instance, they can
simply save the hit for later processing or accumulate them into a histogram.

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

!!! warning
    Currently the Sobol implementation seems to be broken and thus should be
    avoided.
