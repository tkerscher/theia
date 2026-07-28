#include "surface/thin/fresnel_complex.glsl"

//only specular scattering
#define SURFACE_MODEL_SPECULAR

struct SurfaceProperties {
    float reflectance;
    float transmissivity;
    float n_i;
    float n_o;

    //Randomly selected process:
    // 0: absorb
    // 1: reflect
    // 2: transmit
    uint sampledProcess;
};
//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    //fetch optical properties
    uint layerMediumIdx = loadMaterialSlot_uvec2(LAYER_MEDIUM, hit.materialIdx).x;
    float n0 = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, ray.wavelength, 1.0);
    float n1 = lookUpMediaTable1D(REFRACTIVE_INDEX, layerMediumIdx, ray.wavelength, 1.0);
    float k1 = lookUpMediaTable1D(IMAG_REFRACTIVE_INDEX, layerMediumIdx, ray.wavelength, 0.0);
    float n2 = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 1.0);
    float d = loadMaterialConstant(LAYER_THICKNESS, hit.materialIdx);

    //calculate Fresnel coefficients for thin layer
    //fresnel = (Rs, Rp, Ts, Tp)
    float cos0 = abs(dot(ray.direction, hit.rayNrm));
    vec4 fresnel = fresnel_thinLayer(n0, n1, k1, n2, d, ray.wavelength, cos0);
    //for unpolarized light, it's the average
    float R = 0.5 * (fresnel.x + fresnel.y);
    float T = 0.5 * (fresnel.z + fresnel.w);

    //especially for particles it is important to make the same decision in
    //processSurfaceTargetHit() and sampleSurfaceInteraction() to ensure we
    //neither lose nor duplicate a particle
    // -> make decision here
    uint sampledProcess = 0; //default to absorption
    float u = random(idx, dim);
    if (u < R)
        sampledProcess = 1; //reflect
    else if(u < R + T)
        sampledProcess = 2; //transmit
    
    //return properties
    return SurfaceProperties(R, T, n0, n2, sampledProcess);
}

bool processSurfaceTargetHit(
    RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    int objectId,
    out HitItem item,
    uint idx, inout uint dim
) {
    //we ignore the MATERIAL_TRANSMIT_HIT_BIT as it causes problems for evanescent waves
    // bool transmitHit = (hit.flags & MATERIAL_TRANSMIT_HIT_BIT) != 0;

    #ifdef RAY_PARTICLE

    //we can only detect whole particles -> ignore if not absorbed
    if (props.sampledProcess == 0) {
        item = createHit(
            ray,
            hit.objPos,
            hit.objNrm,
            objectId,
            hit.worldToObj
        );
    }
    return props.sampledProcess == 0;

    #else

    //we have a local copy of the ray. Attenuate by absorptivity before detecting
    float A = 1.0 - props.reflectance - props.transmissivity;
    if (A > 0.0) {
        ray.lin_contrib *= A;
        item = createHit(
            ray,
            hit.objPos,
            hit.objNrm,
            objectId,
            hit.worldToObj
        );
    }
    return A > 0.0;

    #endif
}

ResultCode sampleSurfaceInteraction(
    inout RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    uint idx, inout uint dim
) {
    //fetch material flags
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;
    bool canTransmit = (hit.flags & NO_TRANSMIT_BIT) == 0;

    #ifdef RAY_PARTICLE

    //we already importance sampled what to do
    ResultCode result = RESULT_CODE_RAY_ABSORBED;
    if (props.sampledProcess == 1 && canReflect) {
        result = reflectRay(ray, hit);
    }
    else if (props.sampledProcess == 2 && canTransmit) {
        result = transmitRay(ray, hit, props.n_i, props.n_o);
        //due to finite numerical precision, we might run into total internal
        //reflection earlier than expected -> mark as absorbed
        if (result == ERROR_CODE_TOTAL_INTERNAL_REFLECTION)
            result = RESULT_CODE_RAY_ABSORBED;
    }
    //either sampled absorption or sampled process is forbidden
    //-> mark as absorbed (default value)
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #else

    ResultCode result = RESULT_CODE_RAY_ABSORBED;
    if (canReflect && canTransmit) {
        //if we do not want to absorb the ray, sample between reflection and transmission
        uint sampledProcess = props.sampledProcess;
        #ifndef SURFACE_ABSORB_RAY
        float prob_reflect = props.reflectance / (props.reflectance + props.transmissivity);
        sampledProcess = (random(idx, dim) < prob_reflect) ? 1 : 2;
        ray.lin_contrib *= props.reflectance + props.transmissivity; //IS correction
        #endif

        if (sampledProcess == 1) {
            result = reflectRay(ray, hit);
        }
        else if (sampledProcess == 2) {
            result = transmitRay(ray, hit, props.n_i, props.n_o);
            //due to finite numerical precision, we might run into total internal
            //reflection earlier than expected -> mark as absorbed
            if (result == ERROR_CODE_TOTAL_INTERNAL_REFLECTION)
                result = RESULT_CODE_RAY_ABSORBED;
        }
    }
    else if (canReflect && props.reflectance > 0.0) {
        //only reflection allowed -> deterministically reflect
        ray.lin_contrib *= props.reflectance;
        result = reflectRay(ray, hit);
    }
    else if (canTransmit && props.transmissivity > 0.0) {
        //only transmission allowed -> deterministically transmit
        ray.lin_contrib *= props.transmissivity;
        result = transmitRay(ray, hit, props.n_i, props.n_o);
        //due to finite numerical precision, we might run into total internal
        //reflection earlier than expected -> mark as absorbed
        if (result == ERROR_CODE_TOTAL_INTERNAL_REFLECTION)
            result = RESULT_CODE_RAY_ABSORBED;
    }
    
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #endif
}
