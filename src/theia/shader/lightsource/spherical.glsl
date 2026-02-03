#ifndef _INCLUDE_LIGHTSOURCE_SPHERICAL
#define _INCLUDE_LIGHTSOURCE_SPHERICAL

#include "math.glsl"
#include "util/jacobian.glsl"
#include "util/sample.glsl"

uniform LightParams {
    vec3 position;
    uint mediumIdx;
    
    float t_min;
    float t_max;

    float contribFwd;
    float contribBwd;
} lightParams;

ForwardRay sampleLight(uint idx, inout uint dim) {
    //sample wavelength using wavelength source
    #ifdef LIGHT_SOURCE_EMIT_PARTICLE
    float wavelength = sampleWavelength(idx, dim);
    #else
    float contrib;
    float wavelength = sampleWavelength(contrib, idx, dim);
    #endif

    //sample direction
    vec3 direction = sampleUnitSphere(random2D(idx, dim));
    //sample startTime
    //TODO: skip this if ray does not need time
    float v = random(idx, dim);
    float time = mix(lightParams.t_min, lightParams.t_max, v);

    //create forward ray
    return createForwardRay(
        lightParams.position,
        direction,
        wavelength,
        lightParams.mediumIdx,
        time
        #ifndef LIGHT_SOURCE_EMIT_PARTICLE
        , contrib * lightParams.contribFwd
        #endif
    );
}

#ifndef LIGHT_SOURCE_EMIT_PARTICLE

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
    float contrib = lightParams.contribBwd * dw_dA(lightParams.position, observer, normal);

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

#endif

#endif
