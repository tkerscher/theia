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
    //we importance sampled attenuation -> nothing to do
    return RESULT_CODE_SUCCESS;
}

ResultCode sampleVolumeInteraction(
    inout RAY ray,
    uint idx, inout uint dim
) {
    //fetch absorption and scattering coefficient
    float mu_a = lookUpMediaTable1D(ABSORPTION_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    float mu_s = lookUpMediaTable1D(SCATTERING_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    float mu_e = mu_a + mu_s;
    //are there any interactions at all?
    if (mu_e <= 0.0) return RESULT_CODE_SUCCESS; //nothing to do -> carry on

    //if applicable, randomly decide between absorption and scattering
    float p_scatter = mu_s / mu_e;
    #if defined(VOLUME_ABSORB_RAY) || defined(RAY_PARTICLE)
    //stop ray with chance 1-p_scatter
    if (random(idx, dim) > p_scatter) {
        return RESULT_CODE_RAY_ABSORBED;
    }
    #else
    //never absorb -> adjust contrib accordingly
    ray.lin_contrib *= p_scatter;
    #endif

    //scatter ray
    float cos_theta, phi;
    sampleScatterDir(
        ray.mediumIdx,
        ray.direction,
        random2D(idx, dim),
        cos_theta, phi
    );
    vec3 newDir = scatterDir(ray.direction, cos_theta, phi);
    ResultCode result = scatterRay(ray, newDir);
    return result < 0 ? result : RESULT_CODE_RAY_SCATTERED;
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
    ray.log_contrib -= mu_e * dist;
    //if we did not hit anything, prepare interaction sampling
    //this stems from normalizing interaction probabilites
    if (!hit) {
        ray.lin_contrib *= mu_e;
    }

    return RESULT_CODE_SUCCESS;
}

ResultCode volumeScatterRay(
    inout RAY ray,
    vec3 newDir,
    uint idx, inout uint dim
) {
    float mu_a = lookUpMediaTable1D(ABSORPTION_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    float mu_s = lookUpMediaTable1D(SCATTERING_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    float mu_e = mu_a + mu_s;

    ray.lin_contrib *= mu_s / mu_e * scatterProb(ray.mediumIdx, ray.direction, newDir);
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
