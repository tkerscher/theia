#ifndef _INCLUDE_LIGHTSOURCE_SPHERICAL_BACKWARD
#define _INCLUDE_LIGHTSOURCE_SPHERICAL_BACKWARD

#include "util/equiangular.glsl"
#include "util/jacobian.glsl"
#include "lightsource/spherical/common.glsl"

ForwardRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    uint mediumIdx,
    uint idx, inout uint dim
) {
    //get direction
    vec3 direction = normalize(observer - lightParams.position);

    //sample start time
    float u = random(idx, dim);
    float time = mix(lightParams.t_min, lightParams.t_max, u);
    //calculate contribution
    float contrib = lightParams.contribBwd * dw_dA(
        lightParams.position, observer, lightParams.softeningRadius);

    //assemble ray
    return createForwardRay(
        lightParams.position,
        direction,
        wavelength,
        lightParams.mediumIdx, //do not use function parameter here. We might be in a different medium!
        time,
        contrib
    );
}

#define LIGHT_SOURCE_SUPPORTS_MIS
#define LightNeeContext EquiAngularSampleContext

float sampleLightNeeStepSize(
    vec3 origin,
    vec3 direction,
    float t_max,
    float wavelength,
    uint mediumIdx,
    out LightNeeContext context,
    uint idx, inout uint dim
) {
    return sampleEquiAngularStepSize(
        lightParams.position,
        origin,
        direction,
        t_max,
        lightParams.softeningRadius,
        context,
        idx, dim
    );
}

ForwardRay sampleLightMIS(
    vec3 rayPosition,
    vec3 direction,
    float t_step,
    float wavelength,
    uint mediumIdx,
    float residualProb,
    const LightNeeContext context,
    uint idx, inout uint dim
) {
    float contrib = lightParams.contribBwd;
    contrib *= equiAngularMisWeight(
        lightParams.position,
        rayPosition,
        lightParams.softeningRadius,
        t_step,
        residualProb,
        context
    );

    //sample start time
    float u = random(idx, dim);
    float time = mix(lightParams.t_min, lightParams.t_max, u);
    //assemble ray and return
    return createForwardRay(
        lightParams.position,
        normalize(rayPosition - lightParams.position), //TODO: if close causes NaN
        wavelength,
        lightParams.mediumIdx, //do not use function parameter here. We might be in a different medium!
        time,
        contrib
    );
}

#endif
