# Module Overview

Theia is organized in several modules as described in the following table:

| Module | Description |
|--------|-------------|
| [`camera`](camera.md) | Contains definition and implementation of [cameras](../pipeline/components.md#camera). |
| [`cascades`](cascades.md) | Contains parameterization of Cherenkov cascades used by `ParticleCascadeLightSource`. |
| [`estimator`](estimator.md) | Contains definition and implementation of [hit responses](../pipeline/components.md#hit-response) as well as estimator that further process the responses. |
| [`light`](light.md) | Contains definition and implementations of [light sources](../pipeline/components.md#light-source) and [wavelength sources](../pipeline/components.md#wavelength-source). |
| [`lookup`](lookup.md) | Contains logic for creating look up tables as used to store optical properties of media. |
| [`material`](material.md) | Contains definition and implementation of [media](../scene.md#medium) and [materials](../scene.md#materials). |
| [`random`](random.md) | This modules defined and implements [random number generators](../pipeline/components.md#random-number-generator). |
| [`scene`](scene.md) | This module contains code for handling [meshes](../scene.md#meshes-and-volume-border) and creating [scenes](../scene.md). |
| [`target`](target.md) | This module contains the definition of [target](../pipeline/components.md#target) and [target guides](../pipeline/components.md#target-guide) as well as some implementation of them. |
| [`trace`](trace.md) | Contains definition and implementations of [tracer](../pipeline/components.md). |
| [`units`](units.md) | This module defines useful SI units that can be used for e.g. material and tracers. |
