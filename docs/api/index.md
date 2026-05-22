# API Documentation

Theia is split into multiple modules. The following gives a short description
for each of them. See the dedicated pages for the full API documentation of
the respective modules.

- [**`theia.camera`**](camera.md)  
Contains [camera](../pipeline/components.md#camera) components
- [`theia.compiler`](compiler.md)  
Manages loading and compiling of GPU code at runtime
- [`theia.device`](device.md)  
Handles GPU discovery and selection
- [`theia.light`](light.md)  
Contains [light source](../pipeline/components.md#light-source) and
[wavelength source](../pipeline/components.md#wavelength-source) components
- [`theia.lookup`](lookup.md)  
Creating look up tables and uploading them to the GPU
- [`theia.material`](material.md)  
Contains the [medium and material](../scene/material.md)
system
- [`theia.model`](model.md)  
Provides utilities for creating media and materials from analytic optical models.
Also contains some existing optical models.
- [`theia.property`](property.md)  
Contains the optical property system
- [`theia.random`](random.md)  
Contains the [random number generators](../pipeline/components.md#random-number-generator)
- [`theia.ray`](ray.md)
Contains [ray models](../pipeline/components.md#ray-model)
- [`theia.response`](response.md)  
Contains [hit responses](../pipeline/components.md#hit-response)
- [`theia.scene`](scene.md)  
[Mesh](../scene/mesh.md) loading logic and [scene](../scene/scene.md) creation.
- [`theia.surface`](surface.md)  
Defines surface models, including some pre-existing ones
- [`theia.target`](target.md)  
Contains [targets](../pipeline/components.md#target) and
[target guides](../pipeline/components.md#target-guide)
- [`theia.task`](task.md)  
Contains tasks
- [`theia.testing`](testing.md)  
Utility classes used for testing components
- [`theia.trace`](trace.md)  
Contains [tracer](../pipeline/components.md#tracer)
- [`theia.units`](units.md)  
Contains the [units system](../scene/units.md)
- [`theia.volume`](volume.md)  
Defines volume models, including some pre-existing ones
