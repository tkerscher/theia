//only specular scattering
#define SURFACE_MODEL_SPECULAR

struct SurfaceProperties {
    float reflectance;
    float n_i;
    float n_o;
    bool doReflect;
};

//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    //fetch refractive indices and reflectance
    float n_i = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, ray.wavelength, 1.0);
    float n_o = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 1.0);
    float r = lookUpMaterialTable1D(REFLECTIVITY, hit.materialIdx, ray.wavelength, 1.0);

    //especially for particles it is important to make the same decision in
    //processSurfaceTargetHit() and sampleSurfaceInteraction() to ensure we
    //neither lose nor duplicate a particle
    // -> make decision here
    bool doReflect = random(idx, dim) < r;

    //return properties
    return SurfaceProperties(r, n_i, n_o, doReflect);
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

    //we have a local copy of the ray. Attenuate by reflectance before detecting
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

    ResultCode result;
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
    else {
        return RESULT_CODE_RAY_ABSORBED;
    }
    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #endif
}
