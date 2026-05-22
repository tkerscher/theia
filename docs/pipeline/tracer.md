---
icon: lucide/waypoints
---

# Tracer

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

## Scene vs. Volume Tracer

A big difference in tracing is whether one uses a [scene](../scene/scene.md). While
it allows for multiple media and accurate detector models including reflection
and transmission, it also requires ray tracing capable hardware and is generally
computationally more demanding. Alternatively, one can trace a simple homogenous
medium. In that case the tracer requires a suitable component that determines
whether a ray hit the detector.

The distinction between scene and volume tracer can either happen by making the
scene an optional parameter like in direct light tracer
or, if the implementation differs to much, by different tracers distinguished by
name like for `SceneForwardTracer` and `VolumeForwardTracer`.

## Forward Tracer

Forward tracing is perhaps what most people think of when talking about photon
tracing. A light ray is sampled from a [light source](components.md#light-source)
and traced through a [scene](../scene/scene.md) or volume until it hits a target.
The former case is implemented by `SceneForwardTracer` and the latter by
`VolumeForwardTracer`. For volumes the target is defined by a
[Target](components.md#target) component, whereas for scenes it is defined by a
geometry with material that has the `DETECTOR` bit set. If a scene defines
multiple targets, the active one can be selected by specifying the `targetId`
which must match the corresponding `detectorId` to generate hits. Alternatively,
a negative value disables this filtering causing all hits with any detector
material to create a hit.

To increase performance of the simulation both tracers are capable of creating
alternative light paths by deliberately connecting scattering events with the
target. As explained with [path integrals](../model.md#path-integrals), since
these are paths of varying length they contribute to different integrals of
the underlying estimator and are thus safe to use without introducing
auto-correlation. Targets support this out of the box. For scenes one must
additionally provide a [target guide](components.md#target-guide).

## Backward Tracer

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

An odd entry in this category is the `SceneBackwardTargetTracer`: It does not
complete light paths by connecting them to a lightsource, but by intersecting
an object whose material has the `LIGHT_SOURCE` flag set. These intersection are
also used for creating hits instead of the ones sampled by the camera. This
tracer can be used for simulating detector or geometries independent of a light
source.

## Direct Light Tracer

As the name suggest, this tracer estimates the incident light at the detector
that did not undergo any scattering. It does so by first sampling a hit position
from a [camera](components.md#camera) and then the direction of the incident
light from a [light source](components.md#light-source). Both must support this
mode. For incorporating shadowing effects optionally a scene can be specified.

The idea behind this tracer is twofold: It is a very fast estimator and as
direct light often provides the most energy might already be enough. Combined
with other tracer it can also be used to reduce the bias caused by paths these
are missing such as is the case with backward tracing.
