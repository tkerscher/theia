#ifndef _INCLUDE_SURFACE_DIELECTRIC_ROUGH_BACKWARD
#define _INCLUDE_SURFACE_DIELECTRIC_ROUGH_BACKWARD

#include "surface/propagate/backward.glsl"

#define NO_REFLECT_BIT MATERIAL_NO_REFLECT_BWD_BIT
#define NO_TRANSMIT_BIT MATERIAL_NO_TRANSMIT_BWD_BIT

//the model-specific sample_microfacet_normal()/check_microfacet() are prepended
//to this file during compilation (see surface.py)
#define RAY BackwardRay
#include "surface/dielectric_rough/template.glsl"
#undef RAY

#undef NO_REFLECT_BIT
#undef NO_TRANSMIT_BIT

#endif
