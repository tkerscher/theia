#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV_CASCADE_BACKWARD
#define _INCLUDE_LIGHTSOURCE_CHERENKOV_CASCADE_BACKWARD

#include "math.glsl"
#include "random/gamma.glsl"
#include "util/jacobian.glsl"

#include "lightsource/cherenkov/common.glsl"
#include "lightsource/cherenkov/cascade/common.glsl"

ForwardRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    uint mediumIdx,
    uint idx, inout uint dim
) {
    //sample point along shower track
    float z = sampleGamma(cascade.a_long, idx, dim) * cascade.b_long;
    vec3 rayPos = cascade.startPosition + z * cascade.direction;
    float time = cascade.startTime + z * INV_SPEED_OF_LIGHT;

    //fetch refractive index
    float n = lookUpMediaTable1D(REFRACTIVE_INDEX, cascade.mediumIdx, wavelength, 1.0);

    //calculate emission direction
    vec3 rayDir = normalize(observer - rayPos);
    float cos_theta = dot(cascade.direction, rayDir);
    //evaluate emission profile
    float contrib = particle_evalEmissionAngle(
        n, cascade.a_angular, cascade.b_angular, cos_theta);
    //convert integral dA -> dw
    contrib *= abs(cos_theta) * dw_dA(rayPos, observer);
    
    //apply scaling factor
    contrib *= cascade.effectiveLength;
    //if we importance sample the Frank-Tamm formula, we must not apply it here
    //we assume the constant factor was applied elsewhere (e.g. wavelength source)
    #ifndef FRANK_TAMM_IS
    contrib *= frank_tamm(wavelength, n);
    #endif

    //assemble ray
    return createForwardRay(
        rayPos,
        rayDir,
        wavelength,
        cascade.mediumIdx,
        time,
        contrib
    );
}

#endif
