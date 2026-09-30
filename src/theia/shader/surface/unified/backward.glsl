#ifndef _INCLUDE_SURFACE_UNIFIED_BACKWARD
#define _INCLUDE_SURFACE_UNIFIED_BACKWARD

#include "surface/propagate/backward.glsl"

#define NO_REFLECT_BIT MATERIAL_NO_REFLECT_BWD_BIT
#define NO_TRANSMIT_BIT MATERIAL_NO_TRANSMIT_BWD_BIT

//the gaussian micro-facet functions are prepended to this file during
//compilation (see surface.py)
#define RAY BackwardRay
#include "surface/unified/template.glsl"
#undef RAY

#undef NO_REFLECT_BIT
#undef NO_TRANSMIT_BIT

#endif
