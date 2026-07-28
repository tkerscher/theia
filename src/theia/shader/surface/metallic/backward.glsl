#ifndef _INCLUDE_SURFACE_MODEL_METALLIC_BACKWARD
#define _INCLUDE_SURFACE_MODEL_METALLIC_BACKWARD

#include "surface/propagate/backward.glsl"

#define NO_REFLECT_BIT MATERIAL_NO_REFLECT_BWD_BIT
#define NO_TRANSMIT_BIT MATERIAL_NO_TRANSMIT_BWD_BIT

#define RAY BackwardRay
#include "surface/metallic/template.glsl"
#undef RAY

#undef NO_REFLECT_BIT
#undef NO_TRANSMIT_BIT

#endif
