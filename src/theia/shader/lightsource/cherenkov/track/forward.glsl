#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV_TRACK_FORWARD
#define _INCLUDE_LIGHTSOURCE_CHERENKOV_TRACK_FORWARD

#include "math.glsl"

#include "lightsource/cherenkov/common.glsl"
#include "lightsource/cherenkov/track/common.glsl"

ForwardRay sampleLight(uint idx, inout uint dim) {
    //sample wavelength
    #ifdef LIGHT_SOURCE_EMIT_PARTICLE
    float wavelength = sampleWavelength(idx, dim);
    #else
    float contrib;
    float wavelength = sampleWavelength(contrib, idx, dim);
    #endif

    //interpolate track
    float u = random(idx, dim);
    vec3 position = mix(lightParams.trackStart, lightParams.trackEnd, u);
    float startTime = mix(lightParams.startTime, lightParams.endTime, u);

    //fetch refractive index
    float u_lam = normalize_lambda(lightParams.mediumIdx, wavelength);
    float n = lookUpMediaTable1D(REFRACTIVE_INDEX, lightParams.mediumIdx, u_lam, 1.0);

    //sample ray direction
    float cos_theta = 1.0 / n;
    float sin_theta = sqrt(max((1.0 - cos_theta)*(1.0 + cos_theta), 0.0));
    float phi = TWO_PI * random(idx, dim);
    vec3 localDir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    vec3 rayDir = createLocalCOSY(lightParams.trackDir) * localDir;

    //apply frank-tamm and sampling contribution if applicable
    #ifndef LIGHT_SOURCE_EMIT_PARTICLE
    contrib *= TWO_PI * lightParams.trackDist; // 1 / p(x)
    #ifndef FRANK_TAMM_IS
    contrib *= frank_tamm(wavelength, n);
    #endif
    #endif

    //create forward ray
    return createForwardRay(
        position,
        rayDir,
        wavelength,
        lightParams.mediumIdx,
        startTime
        #ifndef LIGHT_SOURCE_EMIT_PARTICLE
        , contrib
        #endif
    );
}

#endif
