#ifndef _INCLUDE_SURFACE_MODEL_ABSORBER_BACKWARD
#define _INCLUDE_SURFACE_MODEL_ABSORBER_BACKWARD

//no scattering
#define SURFACE_MODEL_SPECULAR

ResultCode sampleSurfaceInteraction(
    inout BackwardRay ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    return RESULT_CODE_RAY_ABSORBED;
}

bool processSurfaceTargetHit(
    BackwardRay ray,
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
