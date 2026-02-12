#ifndef _INCLUDE_LIGHTSOURCE_SPHERICAL_FORWARD
#define _INCLUDE_LIGHTSOURCE_SPHERICAL_FORWARD

#include "math.glsl"
#include "util/jacobian.glsl"
#include "util/sample.glsl"

#include "lightsource/spherical/common.glsl"

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

#endif
