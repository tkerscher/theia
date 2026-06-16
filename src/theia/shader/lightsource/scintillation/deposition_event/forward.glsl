#ifndef _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_FORWARD
#define _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_FORWARD

#include "math.glsl"
#include "util/sample.glsl"
#include "lightsource/scintillation/deposition_event/common.glsl"

ForwardRay sampleLight(uint idx, inout uint dim) {

    //sample wavelength using wavelength source
    #ifdef LIGHT_SOURCE_EMIT_PARTICLE
    float wavelength = sampleWavelength(idx, dim);
    #else
    float contrib;
    float wavelength = sampleWavelength(contrib, idx, dim);
    #endif

    //sample position and time along the track segment
    float u = random(idx,dim);
    vec3 position = mix(lightParams.startPosition, lightParams.endPosition, u);
    float time = lightParams.startTime
                     + length(position - lightParams.startPosition) 
                         / max((lightParams.startVelocity + u * (lightParams.endVelocity - lightParams.startVelocity) / 2), 1e-5);

    //sample direction
    vec3 direction = sampleUnitSphere(random2D(idx, dim));

    //sample scintillation time delay from exponential decay
    float v = random(idx, dim);
    float total_time = time - log(1.0 - v) * lightParams.timeConstant;
    
    //calculate contribution
    #ifndef LIGHT_SOURCE_EMIT_PARTICLE
    #ifdef SCINTILLATION_USE_ENERGY
    // To get from photon count to energy in eV, multiply by
    //    h * c             h * c
    // ------------  with  -------- = 1.239841984e-6
    //  e * lambda            e
    contrib *= lightParams.photonYield * lightParams.energyDeposition * 1.239841983e-6 / wavelength;
    #else
    contrib *= lightParams.photonYield * lightParams.energyDeposition;
    #endif
    #endif

    //create forward ray
    return createForwardRay(
        position,
        direction,
        wavelength,
        lightParams.mediumIdx,
        total_time
        #ifndef LIGHT_SOURCE_EMIT_PARTICLE
        , contrib
        #endif
    );
}

#endif
