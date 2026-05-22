---
icon: lucide/mirror-rectangular
---

# Physic Model

Just like [components](components.md), custom [physic models](../scene/material.md#physical-models)
can also be added without touching theia's source code. Again, here we just list
the interfaces the tracers expect the physics model to implement.

## Volume Model

```glsl
--8<-- "src/theia/shader/volume/api.glsl"
```

## Surface Model

```glsl
--8<-- "src/theia/shader/surface/api.glsl"
```

## Property Table

Physics model likely rely on the optical properties stored in the corresponding
property tables. Theia's defines a few utility macros to access them:

```glsl title="material.glsl"
--8<-- "src/theia/shader/material.glsl"
```
