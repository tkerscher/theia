#ifndef _INCLUDE_LIGHTSOURCE_TARGET_DISK
#define _INCLUDE_LIGHTSOURCE_TARGET_DISK

#include "util.sample.glsl"

uniform LightTargetParams {
    vec3 position;
    float radius;
    vec3 normal;

    float contrib;
    mat3 objToWorld;
} lightTargetParams;

float sampleLightTarget(
    float wavelength,
    out vec3 samplePos, out vec3 sampleNrm,
    uint idx, inout uint dim
) {
    //sample point on disk
    vec3 localPos = lightTargetParams.radius * sampleUnitDisk(random2D(idx, dim));
    //transform to world coordinates
    samplePos = lightTargetParams.objToWorld * localPos + lightTargetParams.position;
    sampleNrm = lightTargetParams.normal;

    return lightTargetParams.contrib;
}

#endif
