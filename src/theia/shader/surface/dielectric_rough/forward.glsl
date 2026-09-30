#ifndef _INCLUDE_SURFACE_DIELECTRIC_ROUGH_FORWARD
#define _INCLUDE_SURFACE_DIELECTRIC_ROUGH_FORWARD

#include "surface/propagate/forward.glsl"

#define NO_REFLECT_BIT MATERIAL_NO_REFLECT_FWD_BIT
#define NO_TRANSMIT_BIT MATERIAL_NO_TRANSMIT_FWD_BIT

//the model-specific sample_microfacet_normal()/check_microfacet() are prepended
//to this file during compilation (see surface.py)
#define RAY ForwardRay
#include "surface/dielectric_rough/template.glsl"
#undef RAY

#undef NO_REFLECT_BIT
#undef NO_TRANSMIT_BIT

#endif
