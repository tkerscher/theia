---
icon: lucide/puzzle
---

# Custom Components

Theia's modular approach allows to add new components while reusing old ones
without changing any of theia's source code. You simply have to subclass the
correct [component](../pipeline/components.md) base class and write the
corresponding GPU code.

!!! note
    How to write GPU code specific for `theia` is explained [here](glsl.md)

In theory you are completely free to rewrite all GPU code and only use the
frameworks barebone functionality. In the more likely case of only adding a
few components, you can find the interfaces agreed on by each component in the
following.

## Light Source

```glsl
--8<-- "src/theia/shader/lightsource/api.glsl"
```

## Camera

```glsl
--8<-- "src/theia/shader/camera/api.glsl"
```

## Wavelength Source

```glsl
--8<-- "src/theia/shader/wavelengthsource/api.glsl"
```

## Hit Response

```glsl
--8<-- "src/theia/shader/response/api.glsl"
```

## Target

```glsl
--8<-- "src/theia/shader/target/api.glsl"
```

## Target Guide

```glsl
--8<-- "src/theia/shader/target_guide/api.glsl"
```

```glsl title="target_guide/common.glsl"
--8<-- "src/theia/shader/target_guide/common.glsl"
```
