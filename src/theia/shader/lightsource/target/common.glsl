#ifndef _INCLUDE_LIGHTSOURCE_TARGET_COMMON
#define _INCLUDE_LIGHTSOURCE_TARGET_COMMON

struct LightTargetSample {
    vec3 position;
    vec3 normal;

    uint mediumIdx;

    float contrib;
};

#endif
