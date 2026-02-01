#ifndef _INCLUDE_LIGHTSOURCE_TARGET_DISK
#define _INCLUDE_LIGHTSOURCE_TARGET_DISK

#include "util/sample.glsl"

uniform LightTargetParams {
    vec3 position;
    float radius;
    vec3 normal;

    float contrib;
    uint mediumIdx;

    mat3 objToWorld;
} lightTargetParams;

LightTargetSample sampleLightTarget(
    float wavelength,
    uint idx, inout uint dim
) {
    //sample point on disk
    vec3 localPos = lightTargetParams.radius * sampleUnitDisk(random2D(idx, dim));
    //transform to world coordinates
    vec3 samplePos = lightTargetParams.objToWorld * localPos + lightTargetParams.position;
    vec3 sampleNrm = lightTargetParams.normal;

    return LightTargetSample(
        samplePos,
        sampleNrm,
        lightTargetParams.mediumIdx,
        lightTargetParams.contrib
    );
}

#endif
