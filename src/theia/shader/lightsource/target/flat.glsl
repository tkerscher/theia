#ifndef _INCLUDE_LIGHTSOURCE_TARGET_FLAT
#define _INCLUDE_LIGHTSOURCE_TARGET_FLAT

uniform LightTargetParams {
    float width;
    float height; //length
    vec3 offset;
    vec3 normal;

    float contrib;
    mat3 objToWorld;
} lsTargetParams;

float sampleLightTarget(
    float wavelength,
    out vec3 samplePos, out vec3 sampleNrm,
    uint idx, inout uint dim
) {
    //sample point on flat
    vec2 u = random2D(idx, dim);
    float localX = lsTargetParams.width * (u.x - 0.5);
    float localY = lsTargetParams.height * (u.y - 0.5);
    vec3 localPos = vec3(localX, localY, 0.0);
    //transform to world coordinates
    samplePos = lsTargetParams.objToWorld * localPos + lsTargetParams.offset;
    sampleNrm = lsTargetParams.normal;

    return lsTargetParams.contrib;
}

#endif
