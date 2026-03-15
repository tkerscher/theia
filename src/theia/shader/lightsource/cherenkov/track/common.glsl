#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV_TRACK_COMMON
#define _INCLUDE_LIGHTSOURCE_CHERENKOV_TRACK_COMMON

uniform LightParams {
    vec3 trackStart;
    float startTime;

    vec3 trackEnd;
    float endTime;

    vec3 trackDir;
    float trackDist;

    uint mediumIdx;
} lightParams;

#endif
