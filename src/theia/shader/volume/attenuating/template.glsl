/*
Forward and backward propagation uses the same code, but since ForwardRay and
BackwardRay may behave differently, we need separate functions. Unfortunately,
GLSL does not support templates, so we use macros instead.
*/

#ifndef RAY
#error "RAY template argument not defined"
#endif

#include "volume/scatter.glsl"

float sampleInteractionLength(
    const RAY ray,
    uint idx, inout uint dim
) {
    //fetch absorption and scattering coefficient
    float mu_a = lookUpMediaTable1D(ABSORPTION_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    float mu_s = lookUpMediaTable1D(SCATTERING_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    float mu_e = mu_a + mu_s;

    //sample exponential distribution
    //use u -> 1.0 - u > 0.0 to be safe on the log
    //if mu_e = 0 produces +inf
    float u = random(idx, dim);
    return -log(1.0 - u) / mu_e;
}

ResultCode applyVolumeSampled(
    inout RAY ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
) {
    //we importance sampled the attenuation
    // -> nothing to do if we hit something (no volume interaction happened)
    if (hit) return RESULT_CODE_SUCCESS;

    //fetch absorption and scattering coefficient
    float mu_a = lookUpMediaTable1D(ABSORPTION_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    float mu_s = lookUpMediaTable1D(SCATTERING_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    float mu_e = mu_a + mu_s;
    //chance of scattering
    float p_scatter = mu_s / (mu_a + mu_s);

    #if defined(VOLUME_ABSORB_RAY) || defined(RAY_PARTICLE)
    //stop ray with chance 1-p_scatter
    float u = random(idx, dim);
    if (u > p_scatter) {
        return RESULT_CODE_RAY_ABSORBED;
    }
    #else
    //always scatter. Adjust ray contribution accordingly
    ray.lin_contrib *= p_scatter;
    #endif

    //ray has not been absorbed -> continue
    return RESULT_CODE_SUCCESS;
}

ResultCode sampleVolumeInteraction(
    inout RAY ray,
    uint idx, inout uint dim
) {
    //we only ever scatter -> sample new direction
    //note that the factor mu_s was either importance sampled or applied earlier
    float cos_theta, phi;
    sampleScatterDir(
        ray.mediumIdx,
        ray.direction,
        random2D(idx, dim),
        cos_theta, phi
    );
    vec3 newDir = scatterDir(ray.direction, cos_theta, phi);
    scatterRay(ray, newDir);

    return RESULT_CODE_RAY_SCATTERED;
}

#ifndef RAY_PARTICLE

ResultCode applyVolume(
    inout RAY ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
) {
    //fetch absorption and scattering coefficient
    float mu_a = lookUpMediaTable1D(ABSORPTION_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    float mu_s = lookUpMediaTable1D(SCATTERING_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    float mu_e = mu_a + mu_s;
    
    //attenuate ray
    ray.log_contrib -= (mu_a + mu_s) * dist;
    //if we did not hit anything, we scatter
    if (!hit) {
        ray.lin_contrib *= mu_s;
    }

    return RESULT_CODE_SUCCESS;
}

ResultCode volumeScatterRay(
    inout RAY ray,
    vec3 newDir,
    uint idx, inout uint dim
) {
    ray.lin_contrib *= scatterProb(ray.mediumIdx, ray.direction, newDir);
    return scatterRay(ray, newDir);
}

vec3 sampleVolumeScattering(
    const RAY ray,
    out float prob,
    uint idx, inout uint dim
) {
    float cos_theta, phi;
    prob = sampleScatterDir(
        ray.mediumIdx,
        ray.direction,
        random2D(idx, dim),
        cos_theta, phi
    );
    return scatterDir(ray.direction, cos_theta, phi);
}

float volumeScatterProb(
    const RAY ray,
    vec3 scatteredDir
) {
    return scatterProb(ray.mediumIdx, ray.direction, scatteredDir);
}

#endif
