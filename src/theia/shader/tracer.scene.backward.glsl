//check expected macros
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef PATH_LENGTH
#error "PATH_LENGTH not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "ray.propagate.glsl"
#include "scene.intersect.glsl"
#include "scene.traverse.backward.glsl"

#include "camera.common.glsl"
#include "response.common.glsl"
#include "wavelengthsource.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "photon.glsl"
#include "camera.glsl"
#include "light.glsl"
#include "response.glsl"

#include "callback.util.glsl"

#ifndef DISABLE_DIRECT_LIGHTING
#define USE_SCENE
#include "tracer.direct.common.glsl"
#endif

layout(scalar) uniform DispatchParams {
    uint batchSize;
};

layout(scalar) uniform TraceParams {
    uvec2 camMedium;
    PropagateParams propagation;
} params;

void traceMain() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batchSize)
        return;
    
    //Direct Light Sampling
    #ifndef DISABLE_DIRECT_LIGHTING
    sampleDirect(idx, dim, Medium(params.camMedium), params.propagation);
    uint iPath = 2;
    #else
    uint iPath = 0;
    #endif
    
    //sample camera ray
    Medium medium = Medium(params.camMedium);
    WavelengthSample photon = sampleWavelength(idx, dim);
    CameraRay cam = sampleCameraRay(photon.wavelength, idx, dim);
    BackwardRay ray = createRay(cam, medium, photon);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, iPath++);

    //trace loop
    //one iteration less than PATH_LENGTH as we will always create shadow rays
    //extending the path length by one
    [[unroll]] for (uint i = 1; i < PATH_LENGTH; ++i) {
        //trace ray
        SurfaceHit hit;
        ResultCode result = trace(
            ray, hit,
            idx, dim,
            params.propagation
        );
        if (result >= 0) {
            result = processInteraction(
                ray, hit,
                cam.hit,
                idx, dim,
                params.propagation
            );
        }
        onEvent(ray, result, idx, iPath++);
        //stop codes are negative
        if (result < 0)
            return;
    }

    //finished trace loop, but could go further
    onEvent(ray, RESULT_CODE_MAX_ITER, idx, iPath);
}

void main() {
    initResponse();
    traceMain();
    finalizeResponse();
}
