#ifndef _INCLUDE_TRACER_PROPAGATE_COMMON
#define _INCLUDE_TRACER_PROPAGATE_COMMON

struct PropagationParams {
    //Ensure maxTime is the first member so we can reuse the same binding with
    //generic direct scene tracing code. (a bit hacky but easier to implement)
    float maxTime;
    float maxDist;

    vec3 lowerBBoxCorner;
    vec3 upperBBoxCorner;

    float sampleCoefficient;
};

#endif
