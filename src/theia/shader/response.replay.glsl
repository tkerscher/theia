#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

//check macro settings
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif

layout(local_size_x = BATCH_SIZE) in;

#include "response.queue.glsl"
//test for rare edge case:
//combine HitRecorder & HitReplay but mismatch polarization
//(would require two different version of HitQueue)
#ifdef HIT_QUEUE_POLARIZED
#define REPLAY_HIT_QUEUE_POLARIZED
#undef HIT_QUEUE_POLARIZED
#endif
//user provided reponse function
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
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= hitCount)
        return;
    
    //load hit
    LOAD_HIT(hit, queue, idx)
    //process hit
    response(hit);
}
