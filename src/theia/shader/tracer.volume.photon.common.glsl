#ifndef _INCLUDE_TRACER_VOLUME_PHOTON_COMMON
#define _INCLUDE_TRACER_VOLUME_PHOTON_COMMON

#include "math.glsl"
#include "ray.propagate.glsl"
#include "tracer.photon.queue.glsl"

uniform TraceParams {
    uvec2 medium;
    int objectId;

    PropagateParams propagation;
} params;

//define global medium
#define USE_GLOBAL_MEDIUM
Medium getMedium() {
    return Medium(params.medium);
}
#include "ray.medium.glsl"

#include "ray.response.glsl"
#include "ray.scatter.glsl"
#include "result.glsl"
#include "util.sample.glsl"

#include "response.common.glsl"
#include "target.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "response.glsl"
#include "target.glsl"

#include "callback.util.glsl"

ResultCode trace(
    inout ForwardRay ray,
    uint idx, inout uint dim
) {
    //sample distance
    float u = random(idx, dim);
    float dist = sampleScatterLength(ray, params.propagation, u);

    //trace target
    TargetSample hit = intersectTarget(ray.state.position, ray.state.direction);
    //check if we hit
    bool hitValid = hit.valid && (hit.dist <= dist);

    //update ray
    dist = min(hit.dist, dist);
    ResultCode result = propagateRay(ray, dist, params.propagation);
    updateRayIS(ray, dist, params.propagation, hitValid);
    //check if ray is still in bounds
    if (result < 0)
        return result; //abort tracing
    
    if (hitValid) {
        //Align hit, i.e. rotate polRef to plane of incidence
        alignRayToHit(ray, hit.normal);
        HitItem item = createHit(
            ray,
            hit.objPosition,
            hit.objNormal,
            params.objectId,
            hit.worldToObj
        );
        if (item.contrib > 0.0)
            response(item, idx, dim);

        // return RESULT_CODE_RAY_DETECTED;
        //abort tracing so we do not detect the same photon twice
        return RESULT_CODE_RAY_ABSORBED;
    }
    else {
        //no hit -> scatter
        return RESULT_CODE_RAY_SCATTERED;
    }
}

void traceLoop(ForwardRay ray, uint idx, uint dim, uint pathOffset, bool save) {
    //trace loop
    for (uint i = 1; i <= PATH_LENGTH; ++i) {
        //trace
        ResultCode result = trace(ray, idx, dim);
        //scatter for next iteration
        vec2 u = random2D(idx, dim);
        scatterRay(ray, u);

        //sample absorption throwing away photons early to increase performance
        if (result >= 0) {
            //for correct absorption sampling, we need a normalized stokes vector
            #ifdef POLARIZATION
            ray.state.lin_contrib *= ray.stokes.x;
            ray.stokes /= ray.stokes.x;
            #endif
            //sample absorption
            if (getContrib(ray) <= random(idx, dim)) {
                result = RESULT_CODE_RAY_ABSORBED;
            }
            else {
                //reset contrib
                ray.state.lin_contrib = 1.0;
                ray.state.log_contrib = 0.0;
            }
        }
        onEvent(ray, result, idx, i);
        //stop codes are negative
        if (result < 0)
            return;
    }

    if (save) {
        //save rays and load them in the next iteration to get fully converged work groups again  
        saveRay(ray, idx);
    }
    else {
        //finished tracing, but could go further
        onEvent(ray, RESULT_CODE_MAX_ITER, idx, pathOffset + PATH_LENGTH + 1);
    }
}

#endif
