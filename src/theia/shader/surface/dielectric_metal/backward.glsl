#ifndef _INCLUDE_SURFACE_MODEL_DIELECTRIC_METAL_BACKWARD
#define _INCLUDE_SURFACE_MODEL_DIELECTRIC_METAL_BACKWARD

#include "surface/propagate/backward.glsl"

#define NO_REFLECT_BIT MATERIAL_NO_REFLECT_BWD_BIT
#define NO_TRANSMIT_BIT MATERIAL_NO_TRANSMIT_BWD_BIT

#define RAY BackwardRay
#include "surface/dielectric_metal/template.glsl"
#undef RAY

#undef NO_REFLECT_BIT
#undef NO_TRANSMIT_BIT

#endif
