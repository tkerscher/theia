#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require

//Enable ray tracing
#ifdef USE_SCENE
#extension GL_EXT_ray_query : require
#endif

//Check expected macros
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "ray.glsl"
#include "ray.propagate.glsl"

layout(scalar) uniform TraceParams {
    uvec2 medium;

    PropagateParams propagation;
} params;

#ifdef USE_SCENE

//Top level acceleration structure containing the scene
uniform accelerationStructureEXT tlas;

#endif

//define global medium
#define USE_GLOBAL_MEDIUM
Medium getMedium() {
    return Medium(params.medium);
}
#include "ray.medium.glsl"

#include "ray.combine.glsl"
#include "ray.response.glsl"
#include "result.glsl"

#include "lightsource.common.glsl"
#include "camera.common.glsl"
#include "response.common.glsl"
#include "wavelengthsource.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "camera.glsl"
#include "source.glsl"
#include "response.glsl"
#include "photon.glsl"

#include "callback.util.glsl"

#include "tracer.direct.common.glsl"

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    
    initResponse();
    if (idx < BATCH_SIZE) {
        sampleDirect(idx, dim, Medium(params.medium), params.propagation);
    }
    finalizeResponse();
}
