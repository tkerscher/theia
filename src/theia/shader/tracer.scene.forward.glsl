#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require

//check expected macros
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif
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

layout(scalar) uniform TraceParams {
    uint targetIdx;
    uvec2 sceneMedium;

    PropagateParams propagation;
} params;

//Toggle direct lighting
#ifndef DISABLE_DIRECT_LIGHTING
#define ALLOW_RESPONSE_INIT true
#else
#define ALLOW_RESPONSE_INIT false
#endif

void traceMain() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= BATCH_SIZE)
        return;
    
    //sample ray
    Medium medium = Medium(params.sceneMedium);
    WavelengthSample photon = sampleWavelength(idx, dim);
    ForwardRay ray = createRay(
        sampleLight(photon.wavelength, idx, dim),
        medium, photon);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);

    //trace loop
    bool allowResponse = ALLOW_RESPONSE_INIT;
    [[unroll]] for (uint i = 1; i <= PATH_LENGTH; ++i) {
        //trace ray
        bool last = i == PATH_LENGTH; //mark last trace
        SurfaceHit hit;
        ResultCode result = trace(
            ray, hit,
            params.targetIdx,
            idx, dim,
            params.propagation,
            allowResponse
        );
        if (result >= 0) {
            result = processInteraction(
                ray, hit, params.targetIdx,
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
