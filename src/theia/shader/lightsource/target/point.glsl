#ifndef _INCLUDE_LIGHTSOURCE_TARGET_POINT
#define _INCLUDE_LIGHTSOURCE_TARGET_POINT

uniform LightTargetParams {
    vec3 position;
    uint mediumIdx;
} lightTargetParams;

LightTargetSample sampleLightTarget(
    float wavelength,
    uint idx, inout uint dim
) {
    return LightTargetSample(
        lightTargetParams.position,
        vec3(0.0),
        lightTargetParams.mediumIdx,
        1.0
    );
}

#endif
