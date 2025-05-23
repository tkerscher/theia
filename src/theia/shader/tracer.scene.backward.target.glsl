//check expected macros
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef PATH_LENGTH
#error "PATH_LENGTH not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

#ifdef DISABLE_MIS
#define SCENE_TRAVERSE_DISABLE_MIS
#endif

#include "ray.propagate.glsl"
#define SCENE_TRAVERSE_BACKWARD
#include "scene.traverse.glsl"

#include "camera.common.glsl"
#include "response.common.glsl"
#include "wavelengthsource.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "photon.glsl"
#include "camera.glsl"
#include "response.glsl"

#include "callback.util.glsl"

layout(scalar) uniform DispatchParams {
    uint batchSize;
};

layout(scalar) uniform TraceParams {
    uvec2 camMedium;
    int targetId;
    PropagateParams propagation;
} params;

void traceMain() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batchSize)
        return;
    
    //sample camera ray
    Medium medium = Medium(params.camMedium);
    WavelengthSample photon = sampleWavelength(idx, dim);
    CameraRay cam = sampleCameraRay(photon.wavelength, idx, dim);
    BackwardRay ray = createRay(cam, medium, photon);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);

    //there is no dedicated direct lighting tracer for this setting
    bool allowResponse = true;
    //trace loop
    [[unroll]] for (uint i = 1; i <= PATH_LENGTH; ++i) {
        //trace ray
        bool last = i == PATH_LENGTH;
        SurfaceHit hit;
        ResultCode result = trace(
            ray, hit,
            params.targetId,
            idx, dim,
            params.propagation,
            allowResponse
        );
        if (result >= 0) {
            result = processInteraction(
                ray, hit, params.targetId,
                idx, dim,
                params.propagation,
                allowResponse,
                last
            );
        }
        onEvent(ray, result, idx, i);
        //stop codes are negative
        if (result < 0)
            return;
        
        #ifndef DISABLE_MIS
        //we did MIS on target if we scattered in volume
        // -> disable response for next segment
        allowResponse = result != RESULT_CODE_RAY_SCATTERED;
        #endif
    }

    //finished trace loop, but could go further
    onEvent(ray, RESULT_CODE_MAX_ITER, idx, PATH_LENGTH + 1);
}

void main() {
    initResponse();
    traceMain();
    finalizeResponse();
}
