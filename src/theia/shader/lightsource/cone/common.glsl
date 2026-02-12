#ifndef _INCLUDE_LIGHTSOURCE_CONE_COMMON
#define _INCLUDE_LIGHTSOURCE_CONE_COMMON

uniform LightParams {
    vec3 direction;
    float cosOpeningAngle;
    vec3 position;

    uint mediumIdx;

    float contribFwd;
    float contribBwd;

    float t_min;
    float t_max;

} lightParams;

#endif
