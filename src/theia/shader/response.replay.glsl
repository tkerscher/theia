#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

//check macro settings
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif

layout(local_size_x = BATCH_SIZE) in;

#include "response.queue.glsl"
//user provided reponse function
#include "response.glsl"

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
