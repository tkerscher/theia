#ifndef _INCLUDE_TRACER_PROPAGATE_BACKWARD
#define _INCLUDE_TRACER_PROPAGATE_BACKWARD

#include "tracer/propagate/common.glsl"
#define RAY BackwardRay
#include "tracer/propagate/template.glsl"
#undef RAY

/**
 * Combines the given rays to create a hit. Assumes the forward ray points to
 * the backward ray's current position.
*/
ResultCode combineRays(
    BackwardRay ray,                ///< Ray from camera
    ForwardRay source,              ///< Sampled ray from light source
    const CameraHit cam,            ///< Sampled camera hit
    const PropagationParams params, ///< Propagation params
    out HitItem hit,                ///< Produced hit
    uint idx, inout uint dim        ///< RNG state
) {
    //scatter ray to point towards source
    volumeScatterRay(ray, -source.direction);
    ResultCode result = propagate(
        ray,
        distance(ray.position, source.position),
        false, false,
        params,
        idx, dim
    );
    //return early if propagation failed
    if (result < 0) return result;

    //combine
    hit = combineRaysAligned(source, ray, cam);

    //check time limit if applicable
    #ifdef RAY_TRANSIENT
    if (hit.time > params.maxTime)
        return RESULT_CODE_RAY_DECAYED;
    #endif

    return RESULT_CODE_SUCCESS;
}

#endif
