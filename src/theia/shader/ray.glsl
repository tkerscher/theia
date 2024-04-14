#ifndef _INCLUDE_RAY
#define _INCLUDE_RAY

#include "material.glsl"

struct Ray {
    vec3 position;
    vec3 direction;

#ifndef USE_GLOBAL_MEDIUM //e.g. empty scene tracer
    uvec2 medium;
#endif

    float wavelength;
    float time;
    float lin_contrib;
    float log_contrib;
    MediumConstants constants;
};

#endif
