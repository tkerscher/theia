#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV_MUON_BACKWARD
#define _INCLUDE_LIGHTSOURCE_CHERENKOV_MUON_BACKWARD

#include "math.glsl"
#include "util/jacobian.glsl"
#include "util/sample.glsl"

#include "lightsource/cherenkov/common.glsl"
#include "lightsource/cherenkov/muon/common.glsl"

//TODO: we should importance sample the track to favour positions closer to the
//      observer and nicer viewing angles. Unfortunately this turns out to be
//      tricky if one also considers numerical stability.

ForwardRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    uint mediumIdx,
    uint idx, inout uint dim
) {
    //fetch refractive index
    float n = lookUpMediaTable1D(REFRACTIVE_INDEX, track.mediumIdx, wavelength, 1.0);

    //sample point on track
    float u = random(idx, dim);
    vec3 rayPos = mix(track.startPosition, track.endPosition, u);
    float startTime = mix(track.startTime, track.endTime, u);
    //connect to observer
    vec3 rayDir = normalize(observer - rayPos);
    float cos_theta = dot(rayDir, track.direction);

    //calculate contribution
    float contrib = track.dist * abs(cos_theta) * dw_dA(rayPos, observer); // 1 / p(x)
    contrib *= track.energyScale;
    contrib *= particle_evalEmissionAngle(n, track.a_angular, track.b_angular, cos_theta);
    //apply frank tamm if requested
    #ifndef FRANK_TAMM_IS
    contrib *= frank_tamm(wavelength, n);
    #endif

    //assemble and return ray
    return createForwardRay(
        rayPos,
        rayDir,
        wavelength,
        track.mediumIdx,
        startTime,
        contrib
    );
}

#endif
