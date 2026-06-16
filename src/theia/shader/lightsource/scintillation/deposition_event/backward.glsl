#ifndef _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_BACKWARD
#define _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_BACKWARD

#include "util/jacobian.glsl"
#include "math.glsl"
#include "lightsource/scintillation/deposition_event/common.glsl"

ForwardRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    uint mediumIdx,
    uint idx, inout uint dim
) {

    //sample position and time along the track segment
    float u = random(idx,dim);
    vec3 position = mix(lightParams.startPosition, lightParams.endPosition, u);
    float time = lightParams.startTime
                     + length(position - lightParams.startPosition) 
                         / max((lightParams.startVelocity + u * (lightParams.endVelocity - lightParams.startVelocity) / 2), 1e-5);

    //get direction
    vec3 direction = normalize(observer - position);

    //sample startTime from exponential decay
    float v = random(idx, dim);
    float total_time = time - log(1.0 - v) * lightParams.timeConstant;

    //calculate contribution
    float contrib = dw_dA(position, observer) * INV_4PI;

    #ifdef SCINTILLATION_USE_ENERGY
    // To get from photon count to energy in eV, multiply by
    //    h * c             h * c
    // ------------  with  -------- = 1.239841984e-6
    //  e * lambda            e
    contrib *= lightParams.photonYield * lightParams.energyDeposition * 1.239841983e-6 / wavelength;
    #else
    contrib *= lightParams.photonYield * lightParams.energyDeposition;
    #endif

    //assemble ray
    return createForwardRay(
        position,
        direction,
        wavelength,
        lightParams.mediumIdx, //do not use function parameter here. We might be in a different medium!
        total_time,
        contrib
    );
}

#endif
