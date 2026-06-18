#ifndef _INCLUDE_VOLUME_MODEL_FLUORESCENT_FORWARD
#define _INCLUDE_VOLUME_MODEL_FLUORESCENT_FORWARD

#include "volume/scatter.glsl"
#include "util/sample.glsl"

struct VolumeProperties {
    float mu_a;     ///< absorption
    float mu_s;     ///< scattering
    float mu_f;     ///< fluorescence
};

VolumeProperties prepareVolume(
    const ForwardRay ray
) {
    //look up interaction coefficients
    float mu_a = lookUpMediaTable1D(ABSORPTION_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    float mu_s = lookUpMediaTable1D(SCATTERING_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    float mu_f = lookUpMediaTable1D(FLUORESCENCE_COEF, ray.mediumIdx, ray.wavelength, 0.0);
    //add fluorescence efficiency to absorption
    float qe = loadMediaConstant(FLUORESCENCE_EFFICIENCY, ray.mediumIdx);
    mu_a += mu_f * (1.0 - qe);
    mu_f *= qe;

    return VolumeProperties(mu_a, mu_s, mu_f);
}

void sampleFluorescenceWavelengthShift(
    inout ForwardRay ray,
    uint idx, uint dim
) {
    float u = random(idx, dim);
    
    //optionally, ensure we only emit at longer wavelengths
    #ifdef FLUORESCENCE_NO_WAVELENGTH_UPSHIFT
    float u_lo = lookUpMediaTable1D(FLUORESCENCE_EMISSION_QUANTILE, ray.mediumIdx, ray.wavelength, 0.0);
    u = u * (1.0 - u_lo) + u_lo; //upshift random number to exclude shorter wavelengths
    #endif
    //optionally, apply time delta
    #ifdef RAY_TRANSIENT
    ray.time += loadMediaConstant(FLUORESCENCE_TIME_SHIFT, ray.mediumIdx);
    #endif

    ray.wavelength = lookUpMediaTable1D(FLUORESCENCE_EMISSION_SAMPLING, ray.mediumIdx, u, ray.wavelength);
}

float sampleInteractionLength(
    const ForwardRay ray,
    uint idx, inout uint dim
) {
    const VolumeProperties props = prepareVolume(ray);
    float mu_e = props.mu_a + props.mu_s + props.mu_f;

    //sample exponential distribution
    //use u -> 1.0 - u > 0.0 to be safe on the log
    //if mu_e = 0 produces +inf
    float u = random(idx, dim);
    return -log(1.0 - u) / mu_e;
}

ResultCode applyVolumeSampled(
    inout ForwardRay ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
) {
    //we importance sampled transmission -> nothing to do
    return RESULT_CODE_SUCCESS;
}

ResultCode sampleVolumeInteraction(
    inout ForwardRay ray,
    uint idx, inout uint dim
) {
    const VolumeProperties props = prepareVolume(ray);
    float mu_e = props.mu_a + props.mu_s + props.mu_f;
    if (mu_e == 0.0) return RESULT_CODE_SUCCESS; //no interaction -> keep ray untouched
    if (mu_e <= 0.0) return ERROR_CODE_TRACE_ABORT; //there is no interaction

    //handle absorption
    #if defined(VOLUME_ABSORB_RAY) || defined(RAY_PARTICLE)

    //sample absorption
    float p_absorb = props.mu_a / mu_e;
    if (random(idx, dim) < p_absorb) return RESULT_CODE_RAY_ABSORBED;

    #else

    //never absorb -> adjust contribution accordingly
    ray.lin_contrib *= (props.mu_s + props.mu_f) / mu_e;

    #endif

    //not absorbed -> either bulk scatter or fluorescent scatter
    float p_scatter = props.mu_s / (props.mu_s + props.mu_f);
    vec3 scatteredDir = vec3(0.0);
    vec2 uDir = random2D(idx, dim);
    if (random(idx, dim) < p_scatter) {
        //bulk scatter
        float cos_theta, phi;
        sampleScatterDir(
            ray.mediumIdx,
            ray.direction,
            uDir,
            cos_theta, phi
        );
        scatteredDir = scatterDir(ray.direction, cos_theta, phi);
    }
    else {
        //fluorescent scatter
        scatteredDir = sampleUnitSphere(uDir);

        sampleFluorescenceWavelengthShift(ray, idx, dim);
    }
    //finally, scatter
    ResultCode result = scatterRay(ray, scatteredDir);
    return result < 0 ? result : RESULT_CODE_RAY_SCATTERED;
}

#ifndef RAY_PARTICLE

ResultCode applyVolume(
    inout ForwardRay ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
) {
    //attenuate ray
    const VolumeProperties props = prepareVolume(ray);
    float mu_e = props.mu_a + props.mu_s + props.mu_f;
    ray.log_contrib -= mu_e * dist;

    //if we did not hit anything, prepare interaction sampling
    //this stems from normalizing interaction probabilities
    if (!hit) {
        ray.lin_contrib *= mu_e;
    }

    return RESULT_CODE_SUCCESS;
}

ResultCode volumeScatterRay(
    inout ForwardRay ray,
    vec3 newDir,
    uint idx, inout uint dim
) {
    const VolumeProperties props = prepareVolume(ray);
    if (props.mu_s + props.mu_f <= 0.0) return ERROR_CODE_TRACE_ABORT;

    //we can either scatter via bulk or fluorescence
    float p_scatter = props.mu_s / (props.mu_s + props.mu_f);
    if (random(idx, dim) < p_scatter) {
        ray.lin_contrib *= scatterProb(ray.mediumIdx, ray.direction, newDir);
    }
    else {
        ray.lin_contrib *= INV_4PI; //fluorescence is isotropic
        sampleFluorescenceWavelengthShift(ray, idx, dim);
    }
    //the sampling contribution turns out to be the same in both cases
    float mu_e = props.mu_a + props.mu_s + props.mu_f;
    ray.lin_contrib *= (props.mu_s + props.mu_f) / mu_e;

    //finally, scatter ray
    return scatterRay(ray, newDir);
}

vec3 sampleVolumeScattering(
    const ForwardRay ray,
    out float prob,
    uint idx, inout uint dim
) {
    const VolumeProperties props = prepareVolume(ray);
    if (props.mu_s + props.mu_f <= 0.0) {
        //no scattering possible
        prob = 0.0;
        return vec3(0.0);
    }

    //either sample bulk scattering or fluorescent scattering
    vec3 newDir;
    float pDirScatter;
    vec2 uDir = random2D(idx, dim);
    float p_scatter = props.mu_s / (props.mu_s + props.mu_f);
    if (random(idx, dim) < p_scatter) {
        //bulk scatter
        float cos_theta, phi;
        pDirScatter = sampleScatterDir(
            ray.mediumIdx,
            ray.direction,
            uDir,
            cos_theta, phi
        );
        newDir = scatterDir(ray.direction, cos_theta, phi);
    }
    else {
        //fluorescent scatter
        newDir = sampleUnitSphere(uDir);
        pDirScatter = scatterProb(ray.mediumIdx, ray.direction, newDir);
    }

    //we could have sampled newDir either way -> combine to get sample prob
    prob = p_scatter * pDirScatter + (1.0 - p_scatter) * INV_4PI; //fluorescence is isotropic

    return newDir;
}

float volumeScatterProb(
    const ForwardRay ray,
    vec3 scatteredDir
) {
    const VolumeProperties props = prepareVolume(ray);
    if (props.mu_s + props.mu_f <= 0.0) return 0.0;

    float p_scatter = props.mu_s / (props.mu_s + props.mu_f);
    float pDirScatter = scatterProb(ray.mediumIdx, ray.direction, scatteredDir);

    //we could have sampled newDir either way -> combine to get sample prob
    return p_scatter * pDirScatter + (1.0 - p_scatter) * INV_4PI; //fluorescence is isotropic
}

#endif

#endif
