#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV_MUON_FORWARD
#define _INCLUDE_LIGHTSOURCE_CHERENKOV_MUON_FORWARD

#include "math.glsl"
#include "util/sample.glsl"

#include "lightsource/cherenkov/common.glsl"
#include "lightsource/cherenkov/muon/common.glsl"

ForwardRay sampleLight(uint idx, inout uint dim) {
    //sample wavelength using wavelength source
    #ifdef LIGHT_SOURCE_EMIT_PARTICLE
    float wavelength = sampleWavelength(idx, dim);
    #else
    float contrib;
    float wavelength = sampleWavelength(contrib, idx, dim);
    #endif

    //fetch refractive index
    float n = lookUpMediaTable1D(REFRACTIVE_INDEX, track.mediumIdx, wavelength, 1.0);

    //sample point on track
    float u = random(idx, dim);
    vec3 position = mix(track.startPosition, track.endPosition, u);
    float startTime = mix(track.startTime, track.endTime, u);
    #ifndef LIGHT_SOURCE_EMIT_PARTICLE
    contrib *= track.dist;
    //add secondary particles' light yield by rescaling contrib
    contrib *= track.energyScale;
    #endif

    //sample emission direction
    vec2 v = random2D(idx, dim); //2D for stratification
    float phi = TWO_PI * v.x;
    float cos_theta = particle_sampleEmissionAngle(n, track.a_angular, track.b_angular, v.y);
    //assemble ray direction
    vec3 trackDir = normalize(track.endPosition - track.startPosition);
    vec3 rayDir = createLocalCOSY(trackDir) * sphericalToCartessian(phi, cos_theta);

    //if we importance sample the Frank-Tamm formula, we must not apply it here
    //we assume the constant factor was applied elsewhere (e.g. wavelength source)
    #if !defined(FRANK_TAMM_IS) && !defined(LIGHT_SOURCE_EMIT_PARTICLE)
    contrib *= TWO_PI * frank_tamm(wavelength, n);
    #endif

    //create forward ray
    return createForwardRay(
        position,
        rayDir,
        wavelength,
        track.mediumIdx,
        startTime
        #ifndef LIGHT_SOURCE_EMIT_PARTICLE
        , contrib
        #endif
    );
}

#endif
