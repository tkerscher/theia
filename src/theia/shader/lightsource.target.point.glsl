#ifndef _INCLUDE_LIGHTSOURCE_TARGET_POINT
#define _INCLUDE_LIGHTSOURCE_TARGET_POINT

layout(scalar) uniform LightTargetParams {
    vec3 position;
} lightTargetParams;

float sampleLightTarget(
    float wavelength,
    out vec3 samplePos, out vec3 sampleNrm,
    uint idx, inout uint dim
) {
    samplePos = lightTargetParams.position;
    sampleNrm = vec3(0.0);
    return 1.0;
}

#endif
