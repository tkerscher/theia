#ifndef _INCLUDE_SURFACE_MODEL_ABSORBER_BACKWARD
#define _INCLUDE_SURFACE_MODEL_ABSORBER_BACKWARD

ResultCode sampleSurfaceInteraction(
    inout BackwardRay ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    return RESULT_CODE_RAY_ABSORBED;
}

#endif
