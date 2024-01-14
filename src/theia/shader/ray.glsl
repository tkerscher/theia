#ifndef _INCLUDE_RAY
#define _INCLUDE_RAY

#ifndef N_LAMBDA
#error "N_LAMBDA not defined"
#endif

#include "material.glsl"

struct Sample {
    float wavelength;
    float time;
    float lin_contrib;
    float log_contrib;
    MediumConstants constants;
};

struct Ray {
    vec3 position;
    vec3 direction;

#ifndef USE_GLOBAL_MEDIUM //e.g. empty scene tracer
    uvec2 medium;
#endif

    Sample samples[N_LAMBDA];
};

#endif
