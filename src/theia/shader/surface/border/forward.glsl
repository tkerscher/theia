#ifndef _INCLUDE_SURFACE_MODEL_BORDER_FORWARD
#define _INCLUDE_SURFACE_MODEL_BORDER_FORWARD

//no scattering
#define SURFACE_MODEL_SPECULAR

#include "surface/propagate/forward.glsl"

ResultCode sampleSurfaceInteraction(
    inout ForwardRay ray,
    const SurfaceHit hit,
    uint idx, inout uint normal
) {
    ResultCode result = crossBorder(ray, hit);
    return result >= 0 ? RESULT_CODE_VOLUME_HIT : result;
}

bool processSurfaceTargetHit(
    ForwardRay ray,
    const SurfaceHit hit,
    int objectId,
    out HitItem item,
    uint idx, inout uint dim
) {
    //volume borders cannot be detectors
    return false;
}

#endif
