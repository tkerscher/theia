//check macro settings
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

//optional RNG
#include "rng.glsl"

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
    //init RNG
    uint idx = gl_GlobalInvocationID.x;
    uint dim = 0;
    //init response
    initResponse();

    //process hit
    if (idx < hitCount) {
        LOAD_HIT(hit, queue, idx)
        response(hit, idx, dim);
    }

    //finalize response
    finalizeResponse();
}
