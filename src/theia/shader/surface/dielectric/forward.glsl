#ifndef _INCLUDE_SURFACE_MODEL_DIELECTRIC_FORWARD
#define _INCLUDE_SURFACE_MODEL_DIELECTRIC_FORWARD

#include "surface/propagate/forward.glsl"

#define NO_REFLECT_BIT MATERIAL_NO_REFLECT_FWD_BIT
#define NO_TRANSMIT_BIT MATERIAL_NO_TRANSMIT_FWD_BIT

#define RAY ForwardRay
#include "surface/dielectric/template.glsl"
#undef RAY

bool processSurfaceTargetHit(
    ForwardRay ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    int objectId,
    out HitItem item,
    uint idx, inout uint dim
) {
    //fetch material flags
    bool canTransmit = (hit.flags & NO_TRANSMIT_BIT) == 0;

    //if allowed, we first transmit the ray before detecting
    if (canTransmit) {
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

#undef NO_REFLECT_BIT
#undef NO_TRANSMIT_BIT

#endif
