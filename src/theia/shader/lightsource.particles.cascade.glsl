#ifndef _INCLUDE_LIGHTSOURCE_PARTICLES_CASCADE
#define _INCLUDE_LIGHTSOURCE_PARTICLES_CASCADE

#include "lightsource.common.glsl"
#include "lightsource.particles.common.glsl"
#include "material.glsl"
#include "random.gamma.glsl"
#include "util.sample.glsl"

/**
 * Parameterization of the light yield from a electro-magnetic or hadronic
 * cascade or shower as described in the paper above.
*/
layout(scalar) uniform CascadeParams {
    //geometric properties
    vec3 startPosition;
    float startTime;
    vec3 direction;

    //light yield parameters
    float effectiveLength;
    //angular emission profile
    float a_angular;
    float b_angular;
    //longitudinal profile (Eq. 4.10 in [1])
    float a_long; //gamma dist shape
    float b_long; //gamma dist scale length [m]

    //For better compatability, we use the same defintion of b_long as ice tray.
    //That however means that we combined two things here:
    // - The actual b parameter of the gamma distribution (Eq. 4.10 in [1])
    // - Multiplying the radiation length X_0 to convert cascade depth to actual depth
    //For gamma distributions the following holds:
    // gamma(a, b) = gamma(a, 1) / b
    //If we also multiply X_0, we get ice tray's version of b (called b' in the
    //next line):
    // X_0 * gamma(a, b) = X_0 * gamma(a, 1) / b = b' * gamma(a, 1)
} cascade;

SourceRay sampleLight(
    float wavelength,
    const MediumConstants medium,
    uint idx, inout uint dim
) {
    //sample point along shower track
    float z = sampleGamma(cascade.a_long, idx, dim) * cascade.b_long;
    vec3 rayPos = cascade.startPosition + z * cascade.direction;
    float time = cascade.startTime + z * INV_SPEED_OF_LIGHT;

    //sample emission direction
    vec2 u = random2D(idx, dim); //2D for stratification
    float phi = TWO_PI * u.x;
    float cos_theta = particle_sampleEmissionAngle(
        medium.n, cascade.a_angular, cascade.b_angular, u.y);
    //assemble ray direction
    vec3 rayDir = createLocalCOSY(cascade.direction) * sphericalToCartessian(phi, cos_theta);

    //add secondary particles' light yield by rescaling contrib
    float contrib = cascade.effectiveLength;
    //if we importance sample the Frank-Tamm formula, we must not apply it here
    //we assume the constant factor was applied elsewhere (e.g. wavelength source)
    #ifndef FRANK_TAMM_IS
    contrib *= frank_tamm(wavelength, medium.n);
    #endif

    //return source ray
    return createSourceRay(rayPos, rayDir, time, contrib);
    //NOTE: For now we don't have polarization here
}

SourceRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    const MediumConstants medium,
    uint idx, inout uint dim
) {
    //sample point along shower track
    float z = sampleGamma(cascade.a_long, idx, dim) * cascade.b_long;
    vec3 rayPos = cascade.startPosition + z * cascade.direction;
    float time = cascade.startTime + z * INV_SPEED_OF_LIGHT;

    //calculate emission direction
    vec3 rayDir = normalize(observer - rayPos);
    float cos_theta = dot(cascade.direction, rayDir);
    //evaluate emission profile
    float contrib = particle_evalEmissionAngle(
        medium.n, cascade.a_angular, cascade.b_angular, cos_theta);
    //factor 2pi since we selected a specific direction
    contrib *= INV_2PI;
    //convert integral dA -> dw
    contrib *= dw_dA(rayPos, observer, normal);
    
    //apply scaling factor
    contrib *= cascade.effectiveLength;
    //if we importance sample the Frank-Tamm formula, we must not apply it here
    //we assume the constant factor was applied elsewhere (e.g. wavelength source)
    #ifndef FRANK_TAMM_IS
    contrib *= frank_tamm(wavelength, medium.n);
    #endif

    //return source ray
    return createSourceRay(rayPos, rayDir, time, contrib);    
}

#endif
