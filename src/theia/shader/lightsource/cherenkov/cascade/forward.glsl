#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV_CASCADE_FORWARD
#define _INCLUDE_LIGHTSOURCE_CHERENKOV_CASCADE_FORWARD

#include "math.glsl"
#include "random/gamma.glsl"
#include "util/sample.glsl"

#include "lightsource/cherenkov/common.glsl"
#include "lightsource/cherenkov/cascade/common.glsl"

ForwardRay sampleLight(uint idx, inout uint dim) {
    //sample point along shower track
    float z = sampleGamma(cascade.a_long, idx, dim) * cascade.b_long;
    vec3 rayPos = cascade.startPosition + z * cascade.direction;
    float time = cascade.startTime + z * INV_SPEED_OF_LIGHT;

    //sample wavelength using wavelength source
    #ifdef LIGHT_SOURCE_EMIT_PARTICLE
    float wavelength = sampleWavelength(idx, dim);
    #else
    float contrib;
    float wavelength = sampleWavelength(contrib, idx, dim);
    #endif

    //fetch refractive index
    float u_lam = normalize_lambda(cascade.mediumIdx, wavelength);
    float n = lookUpMediaTable1D(REFRACTIVE_INDEX, cascade.mediumIdx, u_lam, 1.0);

    //sample emission direction
    vec2 u = random2D(idx, dim); //2D for stratification
    float phi = TWO_PI * u.x;
    float cos_theta = particle_sampleEmissionAngle(n, cascade.a_angular, cascade.b_angular, u.y);
    //assemble ray direction
    vec3 rayDir = createLocalCOSY(cascade.direction) * sphericalToCartessian(phi, cos_theta);

    #ifndef LIGHT_SOURCE_EMIT_PARTICLE
    //add secondary particles' light yield by rescaling contrib
    contrib *= cascade.effectiveLength;
    //if we importance sample the Frank-Tamm formula, we must not apply it here
    //we assume the constant factor was applied elsewhere (e.g. wavelength source)
    //apply Frank-Tamm and energy scaling if applicable
    #ifndef FRANK_TAMM_IS
    contrib *= TWO_PI * frank_tamm(wavelength, n);
    #endif
    #endif

    //create forward ray
    return createForwardRay(
        rayPos,
        rayDir,
        wavelength,
        cascade.mediumIdx,
        time
        #ifndef LIGHT_SOURCE_EMIT_PARTICLE
        , contrib
        #endif
    );
}

#endif
