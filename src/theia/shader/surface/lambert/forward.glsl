#ifndef _INCLUDE_SURFACE_MODEL_LAMBERT_FORWARD
#define _INCLUDE_SURFACE_MODEL_LAMBERT_FORWARD

#include "surface/propagate/forward.glsl"

#define NO_REFLECT_BIT MATERIAL_NO_REFLECT_FWD_BIT
#define NO_TRANSMIT_BIT MATERIAL_NO_TRANSMIT_FWD_BIT

#define RAY ForwardRay
#include "surface/lambert/template.glsl"
#undef RAY

#undef NO_REFLECT_BIT
#undef NO_TRANSMIT_BIT

#endif
