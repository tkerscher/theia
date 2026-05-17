#include "util/sample.glsl"

struct SurfaceProperties {
    float reflectance;

    bool canReflect;
    bool doReflect;
};

//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    //fetch surface properties
    float r = lookUpMaterialTable1D(REFLECTIVITY, hit.materialIdx, ray.wavelength, 1.0);
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;

    //especially for particles it is important to make the same decision in
    //processSurfaceTargetHit() and sampleSurfaceInteraction() to ensure we
    //neither lose nor duplicate a particle
    bool doReflect = random(idx, dim) < r;

    return SurfaceProperties(r, canReflect, doReflect);
}

bool processSurfaceTargetHit(
    RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    int objectId,
    out HitItem item,
    uint idx, inout uint dim
) {
    #ifdef RAY_PARTICLE
    
    if (!props.doReflect) {
        item = createHit(
            ray,
            hit.objPos,
            hit.objNrm,
            objectId,
            hit.worldToObj
        );
    }

    //we can only detect whole particles -> ignore if we sampled reflection earlier
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
    //sample new direction
    vec3 localDir = sampleHemisphereCosine(random2D(idx, dim));
    vec3 rayDir = createLocalCOSY(hit.rayNrm) * localDir;

    ResultCode result;
    #ifdef RAY_PARTICLE

    if (props.doReflect && props.canReflect) {
        result = reflectRay(ray, hit, rayDir);
    }
    else {
        result = RESULT_CODE_RAY_ABSORBED;
    }

    #else

    if (props.canReflect) {
        ray.lin_contrib *= props.reflectance;
        result = reflectRay(ray, hit, rayDir);
    }
    else {
        result = RESULT_CODE_RAY_ABSORBED;
    }

    #endif

    //success?
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;
}

#ifndef RAY_PARTICLE

ResultCode surfaceScatterRay(
    inout RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    vec3 newDir,
    uint idx, inout uint dim
) {
    //we can only reflect
    if (
        !props.canReflect ||
        dot(ray.direction, hit.rayNrm) >= 0.0 ||
        dot(newDir, hit.rayNrm) < 0.0
    )
        return RESULT_CODE_RAY_ABSORBED;
    
    //attenuate by reflectivity
    ray.lin_contrib *= props.reflectance;
    //apply phase function
    ray.lin_contrib *= INV_PI;

    //reflect
    return reflectRay(ray, hit, newDir);
}

vec3 sampleSurfaceScattering(
    const RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    out float prob,
    uint idx, inout uint dim
) {
    vec3 localDir = sampleHemisphereCosine(random2D(idx, dim));
    vec3 rayDir = createLocalCOSY(hit.rayNrm) * localDir;
    
    prob = sampleHemisphereCosinePdf(localDir);

    return rayDir;
}

float surfaceScatterProb(
    RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    vec3 scatteredDir
) {
    //we can only sample reflected reflected directions
    //since we need the cosine anyway, we can clamp it to zero at the bottom
    //to set non-reflected direction to zero probability
    //
    // p(cos theta) = cos(theta) / pi    if cos(theta) > 0
    
    return max(0.0, dot(scatteredDir, hit.rayNrm)) * INV_PI;
}

#endif
