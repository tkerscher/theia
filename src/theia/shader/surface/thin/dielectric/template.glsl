#define SURFACE_MODEL_SPECULAR

struct SurfaceProperties {
    float reflectance;
    float n_i;
    float n_o;
    bool doReflect;
};
//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

//whether to include interference effects present in very thin layers
//with thickness comparable to the wavelength
#ifdef SURFACE_THIN_LAYER_INCLUDE_INTERFERENCE
#include "surface/thin/fresnel_complex.glsl"
#else
#include "surface/thin/fresnel_real.glsl"
#endif

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    //fetch optical properties
    uint layerMediumIdx = loadMaterialSlot_uvec2(LAYER_MEDIUM, hit.materialIdx).x;
    float n0 = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, ray.wavelength, 1.0);
    float n1 = lookUpMediaTable1D(REFRACTIVE_INDEX, layerMediumIdx, ray.wavelength, 1.0);
    float n2 = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 1.0);
    #ifdef SURFACE_THIN_LAYER_INCLUDE_INTERFERENCE
    float d = loadMaterialConstant(LAYER_THICKNESS, hit.materialIdx);
    #endif

    //calculate Fresnel coefficients for thin layer
    //fresnel = (Rs, Rp, Ts, Tp)
    float cos0 = abs(dot(ray.direction, hit.rayNrm));
    #ifdef SURFACE_THIN_LAYER_INCLUDE_INTERFERENCE
    vec4 fresnel = fresnel_thinLayer(n0, n1, 0.0, n2, d, ray.wavelength, cos0);
    #else
    vec4 fresnel = fresnel_thinLayer(n0, n1, n2, cos0);
    #endif
    //for unpolarized light, it's the average
    float R = 0.5 * (fresnel.x + fresnel.y);
    //we expect no absorption -> T = 1 - R

    //especially for particles it is important to make the same decision in
    //processSurfaceTargetHit() and sampleSurfaceInteraction() to ensure we
    //neither lose nor duplicate a particle
    // -> make decision here
    bool doReflect = random(idx, dim) < R;

    //return properties
    return SurfaceProperties(R, n0, n2, doReflect);
}

bool processSurfaceTargetHit(
    RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    int objectId,
    out HitItem item,
    uint idx, inout uint dim
) {
    //if requested, transmit ray before detecting
    //(this does not change ray contribution)
    bool transmitHit = (hit.flags & MATERIAL_TRANSMIT_HIT_BIT) != 0;
    if (transmitHit) {
        transmitRay(ray, hit);
    }

    #ifdef RAY_PARTICLE

    //we can only detect whole particles -> ignore if we sampled reflection earlier
    if (!props.doReflect) {
        item = createHit(
            ray,
            hit.objPos,
            hit.objNrm,
            objectId,
            hit.worldToObj
        );
    }
    return !props.doReflect;

    #else

    //we have a local copy of the ray. Attenuate by transmittance before detecting
    ray.lin_contrib *= (1.0 - props.reflectance);
    item = createHit(
        ray,
        hit.objPos,
        hit.objNrm,
        objectId,
        hit.worldToObj
    );
    return true;

    #endif
}

ResultCode sampleSurfaceInteraction(
    inout RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    uint idx, inout uint dim
) {
    //fetch material flags
    bool isDetector = (hit.flags & MATERIAL_DETECTOR_BIT) != 0;
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;
    bool canTransmit = (hit.flags & NO_TRANSMIT_BIT) == 0;
    //detector implies no transmit
    canTransmit = !isDetector && canTransmit;

    //handle surface reflection/transmission
    #ifdef RAY_PARTICLE

    //importance sample what to do
    ResultCode result;
    if (props.doReflect && canReflect) {
        result = reflectRay(ray, hit);
    }
    else if (!props.doReflect && canTransmit) {
        result = transmitRay(ray, hit, props.n_i, props.n_o);
        //due to finite numerical precision, we might run into total internal
        //reflection earlier than expected -> mark as absorbed
        if (result == ERROR_CODE_TOTAL_INTERNAL_REFLECTION)
            result = RESULT_CODE_RAY_ABSORBED;
    }
    else {
        //sampled decision is forbidden -> abort tracing
        //TODO: Use a better code here?
        return RESULT_CODE_RAY_ABSORBED;
    }
    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #else

    ResultCode result = RESULT_CODE_RAY_ABSORBED;
    if (canReflect && canTransmit) {
        if (props.doReflect) {
            result = reflectRay(ray, hit);
        }
        else {
            result = transmitRay(ray, hit, props.n_i, props.n_o);
        }
    }
    else if(canReflect && props.reflectance > 0.0) {
        //only reflection allowed -> deterministically reflect
        ray.lin_contrib *= props.reflectance;
        result = reflectRay(ray, hit);
    }
    else if(canTransmit && props.reflectance < 1.0) {
        //only transmission allowed -> deterministically transmit
        ray.lin_contrib *= (1.0 - props.reflectance);
        result = transmitRay(ray, hit, props.n_i, props.n_o);
        //due to finite numerical precision, we might run into total internal
        //reflection earlier than expected -> mark as absorbed
        if (result == ERROR_CODE_TOTAL_INTERNAL_REFLECTION)
            result = RESULT_CODE_RAY_ABSORBED;
    }
    //in all other case we keep the default decision "absorb"

    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #endif
}
