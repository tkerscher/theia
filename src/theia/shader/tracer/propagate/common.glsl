#ifndef _INCLUDE_TRACER_PROPAGATE_COMMON
#define _INCLUDE_TRACER_PROPAGATE_COMMON

struct PropagationParams {
    float sampleCoefficient;

    vec3 lowerBBoxCorner;
    vec3 upperBBoxCorner;

    float maxTime;
    float maxDist;
};

#endif
