#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV_TRACK_BACKWARD
#define _INCLUDE_LIGHTSOURCE_CHERENKOV_TRACK_BACKWARD

#include "math.glsl"

#include "lightsource/cherenkov/common.glsl"
#include "lightsource/cherenkov/track/common.glsl"

ForwardRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    uint mediumIdx,
    uint idx, inout uint dim
) {
    //fetch refractive index
    float u_lam = normalize_lambda(lightParams.mediumIdx, wavelength);
    float n = lookUpMediaTable1D(REFRACTIVE_INDEX, lightParams.mediumIdx, u_lam, 1.0);
    //calculate cherenkov angle
    float cos_theta = 1.0 / n;
    float sin_theta = sqrt(max((1.0 - cos_theta) * (1.0 + cos_theta), 0.0));

    //get point on track closest to observer
    float mu = dot(observer - lightParams.trackStart, lightParams.trackDir);
    vec3 C = lightParams.trackStart + mu * lightParams.trackDir;
    float d = distance(observer, C);
    mu -= cos_theta / sin_theta * d;
    float u = mu / lightParams.trackDist;
    //get point of light source
    vec3 position = mix(lightParams.trackStart, lightParams.trackEnd, u);
    float startTime = mix(lightParams.startTime, lightParams.endTime, u);
    vec3 rayDir = normalize(observer - position);

    //calculate contribution
    float contrib = abs(cos_theta) / distance(observer, position);
    //set contrib to zero if light source is not on track
    contrib *= float(mu >= 0.0 && mu <= lightParams.trackDist);
    //apply frank tamm if requested
    #ifndef FRANK_TAMM_IS    
    contrib *= frank_tamm(wavelength, n);
    #endif

    //assemble ray
    return createForwardRay(
        position,
        rayDir,
        wavelength,
        lightParams.mediumIdx, //do not use function parameter here. We might be in a different medium!
        startTime,
        contrib
    );
}

#endif
