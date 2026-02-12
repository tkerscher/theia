#ifndef _INCLUDE_LIGHTSOURCE_SPHERICAL_COMMON
#define _INCLUDE_LIGHTSOURCE_SPHERICAL_COMMON

uniform LightParams {
    vec3 position;
    uint mediumIdx;
    
    float t_min;
    float t_max;

    float contribFwd;
    float contribBwd;
} lightParams;

#endif
