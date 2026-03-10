#ifndef _INCLUDE_SURFACE_MODEL_BORDER_BACKWARD
#define _INCLUDE_SURFACE_MODEL_BORDER_BACKWARD

#include "surface/propagate/backward.glsl"

ResultCode sampleSurfaceInteraction(
    inout BackwardRay ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    ResultCode result = crossBorder(ray, hit);
    return result >= 0 ? RESULT_CODE_VOLUME_HIT : result;
}

bool processSurfaceTargetHit(
    BackwardRay ray,
    const SurfaceHit hit,
    int objectId,
    out HitItem item,
    uint idx, inout uint dim
) {
    //volume borders cannot be detectors
    return false;
}

#endif
