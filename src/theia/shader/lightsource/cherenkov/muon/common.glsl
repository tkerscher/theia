#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV_MUON_COMMON
#define _INCLUDE_LIGHTSOURCE_CHERENKOV_MUON_COMMON

uniform MuonTrackParams {
    //geometric properties
    vec3 startPosition;
    float startTime;
    vec3 endPosition;
    float endTime;
    vec3 direction;
    float dist;

    //light yield parameters
    float energyScale;
    //angular distribution parameters
    //see notebooks/track_angular_dist_fit.ipynb
    float a_angular;
    float b_angular;

    uint mediumIdx;
} track;

#endif
