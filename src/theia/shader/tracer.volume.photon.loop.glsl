//check expected macros
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef PATH_LENGTH
#error "PATH_LENGTH not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "tracer.volume.photon.common.glsl"

layout(scalar, push_constant) uniform Push {
    uint pathOffset;
    uint dim;
    uint save;
} push;

void traceMain() {
    uint idx = gl_GlobalInvocationID.x;
    uint dim = push.dim;
    if (idx >= photonQueueIn.count)
        return;
    
    //load ray
    ForwardRay ray = loadRay(idx);
    idx = photonQueueIn.queue.idx[idx];

    //trace loop
    traceLoop(ray, idx, dim, push.pathOffset, push.save != 0);
}

void main() {
    initResponse();
    traceMain();
    finalizeResponse();
}
