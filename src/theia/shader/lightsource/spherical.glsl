#ifndef _INCLUDE_LIGHTSOURCE_SPHERICAL
#define _INCLUDE_LIGHTSOURCE_SPHERICAL

#include "math.glsl"
#include "util.sample.glsl"

uniform LightParams {
    vec3 position;
    float contribFwd;
    float contribBwd;

    float t_min;
    float t_max;
} lightParams;

SourceRay sampleLight(
    float wavelength,
    const MediumConstants medium,
    uint idx, inout uint dim
) {
    //sample direction
    vec3 rayDir = sampleUnitSphere(random2D(idx, dim));
    //sample startTime
    float v = random(idx, dim);
    float startTime = mix(lightParams.t_min, lightParams.t_max, v);

    //assemble source ray
    return createSourceRay(
        lightParams.position,
        rayDir,
        startTime,
        lightParams.contribFwd
    );
}

SourceRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    const MediumConstants medium,
    uint idx, inout uint dim
) {
    //get direction
    vec3 direction = normalize(observer - lightParams.position);
    
    //sample start time
    float u = random(idx, dim);
    float startTime = mix(lightParams.t_min, lightParams.t_max, u);
    //calculate contribution
    float contrib = lightParams.contribBwd * dw_dA(lightParams.position, observer, normal);

    //assemble source ray
    return createSourceRay(
        lightParams.position,
        direction,
        startTime,
        contrib
    );
}

#endif
