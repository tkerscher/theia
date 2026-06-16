#ifndef _INCLUDE_SURFACE_MODEL_BECKMANN_BACKWARD
#define _INCLUDE_SURFACE_MODEL_BECKMANN_BACKWARD

#include "surface/propagate/backward.glsl"

#define NO_REFLECT_BIT MATERIAL_NO_REFLECT_BWD_BIT
#define NO_TRANSMIT_BIT MATERIAL_NO_TRANSMIT_BWD_BIT

#define RAY BackwardRay
#include "surface/dielectric_rough/beckmann/template.glsl"
#include "surface/dielectric_rough/common.glsl"
#undef RAY

#undef NO_REFLECT_BIT
#undef NO_TRANSMIT_BIT

#endif
