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
#ifndef DIM_OFFSET
#error "DIM_OFFSET not defined"
#endif
// #samples per iteration
#ifdef DISABLE_MIS
#define SCENE_TRAVERSE_DISABLE_MIS 1
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "ray.propagate.glsl"
#include "ray.sample.glsl"
#include "scene.intersect.glsl"
#include "scene.traverse.glsl"

#include "lightsource.common.glsl"
#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "source.glsl"
#include "response.glsl"

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

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= BATCH_SIZE)
        return;
    
    //sample ray
    Medium medium = Medium(params.sceneMedium);
    Ray ray = createRay(sampleLight(idx), medium);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);
    //advange rng by amount used by sampleLight()
    uint dim = DIM_OFFSET;

    //trace loop
    bool allowResponse = ALLOW_RESPONSE_INIT;
    [[unroll]] for (uint i = 1; i <= PATH_LENGTH; ++i, dim += SCENE_TRAVERSE_TRACE_RNG_STRIDE) {
        //trace ray
        bool last = i == PATH_LENGTH; //mark last trace
        ResultCode result = trace(
            ray, params.targetIdx,
            idx, dim,
            params.propagation,
            allowResponse, last);
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
}
