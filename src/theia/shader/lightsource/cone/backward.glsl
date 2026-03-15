#ifndef _INCLUDE_LIGHTSOURCE_CONE_FORWARD
#define _INCLUDE_LIGHTSOURCE_CONE_FORWARD

#include "util/jacobian.glsl"
#include "lightsource/cone/common.glsl"

ForwardRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    uint mediumIdx,
    uint idx, inout uint dim
) {
    //get direction
    vec3 rayDir = normalize(observer - lightParams.position);    
    //calculate contribution (zero if outside cone)
    float cos_angle = dot(rayDir, lightParams.direction);
    float contrib = lightParams.contribBwd * float(cos_angle > lightParams.cosOpeningAngle);
    contrib *= dw_dA(lightParams.position, observer);
    //sample start time
    float u = random(idx, dim);
    float startTime = mix(lightParams.t_min, lightParams.t_max, u);

    //assemble forward ray
    return createForwardRay(
        lightParams.position,
        rayDir,
        wavelength,
        lightParams.mediumIdx,
        startTime,
        contrib
    );
}

#endif
