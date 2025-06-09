//check expected macros
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef PATH_LENGTH
#error "PATH_LENGTH not defined"
#endif

//configure tracing
#define SCENE_TRAVERSE_DISABLE_MIS 1
#define SCENE_TRAVERSE_FORWARD 1

layout(local_size_x = BLOCK_SIZE) in;

#include "ray.propagate.glsl"
#include "scene.intersect.glsl"
#include "scene.traverse.glsl"
#include "tracer.photon.queue.glsl"

#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "response.glsl"

#include "callback.util.glsl"

layout(scalar) uniform TraceParams {
    int targetId;
    uvec2 sceneMedium;

    PropagateParams propagation;
} params;

layout(scalar, push_constant) uniform Push {
    uint pathOffset;
    uint dim;
    uint save;
} push;

void traceMain() {
    uint idx = gl_GlobalInvocationID.x;
    uint dim = push.dim;
    if (idx >= photonQueueIn.count)
        return;
    
    //load ray
    ForwardRay ray = loadRay(idx);
    idx = photonQueueIn.queue.idx[idx];

    //trace loop
    for (uint i = 1; i <= PATH_LENGTH; ++i) {
        //trace ray
        SurfaceHit hit;
        ResultCode result = trace(
            ray, hit,
            params.targetId,
            idx, dim,
            params.propagation,
            true //allow response
        );
        if (result >= 0) {
            result = processInteraction(
                ray, hit, params.targetId,
                idx, dim,
                params.propagation,
                true, //allow response
                false //last
            );
        }

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

        onEvent(ray, result, idx, i + push.pathOffset);
        //stop codes are negative
        if (result < 0)   
            return;
    }

    if (push.save != 0) {
        //save rays and load them in the next iteration to get fully converged work groups again  
        saveRay(ray, idx);
    }
    else {
        //finished tracing, but could go further
        onEvent(ray, RESULT_CODE_MAX_ITER, idx, push.pathOffset + PATH_LENGTH + 1);
    }
}

void main() {
    initResponse();
    traceMain();
    finalizeResponse();
}
