//check expected macros
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef PATH_LENGTH
#error "PATH_LENGTH not defined"
#endif
// #samples per iteration
#ifdef DISABLE_MIS
#define SCENE_TRAVERSE_DISABLE_MIS 1
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "ray.propagate.glsl"
#include "scene.intersect.glsl"
#define SCENE_TRAVERSE_FORWARD
#include "scene.traverse.glsl"

#include "wavelengthsource.common.glsl"
#include "lightsource.common.glsl"
#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "source.glsl"
#include "photon.glsl"
#include "response.glsl"

#include "callback.util.glsl"

layout(scalar) uniform DispatchParams {
    uint batchSize;
};

layout(scalar) uniform TraceParams {
    int targetId;
    uvec2 sceneMedium;

    PropagateParams propagation;
} params;

//Toggle direct lighting
#ifndef DISABLE_DIRECT_LIGHTING
#define ALLOW_RESPONSE_INIT true
#else
#define ALLOW_RESPONSE_INIT false
#endif

ForwardRay sampleRay(uint idx, inout uint dim) {
    WavelengthSample photon = sampleWavelength(idx, dim);
    Medium medium = Medium(params.sceneMedium);
    MediumConstants constants = lookUpMedium(medium, photon.wavelength);
    SourceRay lightRay = sampleLight(photon.wavelength, constants, idx, dim);
    return createRay(lightRay, medium, constants, photon);
}

void traceMain() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batchSize)
        return;
    
    //sample ray
    ForwardRay ray = sampleRay(idx, dim);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);

    //trace loop
    bool allowResponse = ALLOW_RESPONSE_INIT;
    [[unroll]] for (uint i = 1; i <= PATH_LENGTH; ++i) {
        //trace ray
        bool last = i == PATH_LENGTH; //mark last trace
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
#else
        allowResponse = true;
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
