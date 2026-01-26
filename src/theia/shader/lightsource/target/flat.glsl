#ifndef _INCLUDE_LIGHTSOURCE_TARGET_FLAT
#define _INCLUDE_LIGHTSOURCE_TARGET_FLAT

uniform LightTargetParams {
    float width;
    float height; //length
    vec3 offset;
    vec3 normal;

    float contrib;
    uint mediumIdx;

    mat3 objToWorld;
} lightTargetParams;

LightTargetSample sampleLightTarget(
    float wavelength,
    uint idx, inout uint dim
) {
    //sample point on flat
    vec2 u = random2D(idx, dim);
    float localX = lightTargetParams.width * (u.x - 0.5);
    float localY = lightTargetParams.height * (u.y - 0.5);
    vec3 localPos = vec3(localX, localY, 0.0);
    //transform to world coordinates
    vec3 samplePos = lightTargetParams.objToWorld * localPos + lightTargetParams.offset;
    vec3 sampleNrm = lightTargetParams.normal;

    return LightTargetSample(
        samplePos,
        sampleNrm,
        lightTargetParams.mediumIdx,
        lightTargetParams.contrib
    );
}

#endif
