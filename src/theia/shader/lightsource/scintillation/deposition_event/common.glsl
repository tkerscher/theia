#ifndef _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_COMMON
#define _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_COMMON

uniform LightParams {
    vec3 startPosition;
    vec3 endPosition;

    uint mediumIdx;
    
    float startTime;
    float startVelocity;
    float endVelocity;

    float energyDeposition;
    float photonYield;
    float timeConstant;
} lightParams;

#endif
