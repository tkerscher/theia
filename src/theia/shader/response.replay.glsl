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

//check macro settings
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "response.queue.glsl"
//test for rare edge case:
//combine HitRecorder & HitReplay but mismatch polarization
//(would require two different version of HitQueue)
#ifdef HIT_QUEUE_POLARIZED
#define REPLAY_HIT_QUEUE_POLARIZED
#undef HIT_QUEUE_POLARIZED
#endif
//user provided response function
#include "response.glsl"

//check for queue mismatch
#ifdef _INCLUDE_RESPONSE_RECORD
#if defined(REPLAY_HIT_QUEUE_POLARIZED) != defined(HIT_QUEUE_POLARIZED)
#error "mismatch in hit queue definition"
#endif
#endif

//input queue
layout(scalar) readonly buffer HitQueueIn {
    uint hitCount;
    HitQueue queue;
};

void main() {
    //init response
    initResponse();

    //process hit
    uint idx = gl_GlobalInvocationID.x;
    if (idx < hitCount) {
        LOAD_HIT(hit, queue, idx)
        response(hit);
    }

    //finalize response
    finalizeResponse();
}
