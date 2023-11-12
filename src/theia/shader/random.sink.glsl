#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

#include "rng.glsl"

//code dimension
#ifndef PARALLEL_STREAMS
#define PARALLEL_STREAMS 32
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 4
#endif
#ifndef DRAWS
#define DRAWS 16
#endif

layout(local_size_x = PARALLEL_STREAMS, local_size_y = BATCH_SIZE) in;

layout(scalar) writeonly buffer RngSink {
    float u[];
};

layout(scalar) uniform Params {
    uint baseStream;
    uint baseCount;

    uint streams;
    uint samples;
};

//Drawing numbers is consecutively faster (caching),
//but saving them to memory is strided faster
// -> buffer the result in shared memory
shared float u_shared[PARALLEL_STREAMS][BATCH_SIZE * DRAWS];

void main() {
    uint stream = baseStream + gl_GlobalInvocationID.x;
    uint bufStream = gl_LocalInvocationID.x;

    uint count = baseCount + gl_GlobalInvocationID.y * DRAWS;
    uint bufIdx = gl_LocalInvocationID.y * DRAWS;

    for (int i = 0; i < DRAWS; ++i, ++bufIdx, ++count) {
        u_shared[bufStream][bufIdx] = random(stream, count);
    }

    //barrier to ensure local buffer finished
    memoryBarrierShared();
    barrier();

    //discard streams that are not needed
    //earlier not possible due to barrier()
    if (gl_GlobalInvocationID.x > streams)
        return;

    //stride write to memory
    uint streamIdx = gl_GlobalInvocationID.x * samples;
    uint offset = gl_WorkGroupID.y * DRAWS * BATCH_SIZE + gl_LocalInvocationID.y;
    uint idx = streamIdx + offset;
    bufIdx = gl_LocalInvocationID.y;
    for (
        uint i = 0;
        i < DRAWS && offset < samples;
        ++i, idx += BATCH_SIZE, offset += BATCH_SIZE, bufIdx += BATCH_SIZE
    ) {
        u[idx] = u_shared[bufStream][bufIdx];
    }
}
