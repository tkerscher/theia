#ifndef _INCLUDE_SURFACE_MODEL_ABSORBER_FORWARD
#define _INCLUDE_SURFACE_MODEL_ABSORBER_FORWARD

ResultCode sampleSurfaceInteraction(
    inout ForwardRay ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    return RESULT_CODE_RAY_ABSORBED;
}

bool processSurfaceTargetHit(
    ForwardRay ray,
    const SurfaceHit hit,
    int objectId,
    out HitItem item,
    uint idx, inout uint dim
) {
    item = createHit(
        ray,
        hit.objPos,
        hit.objNrm,
        objectId,
        hit.worldToObj
    );
    return true;
}

#endif
