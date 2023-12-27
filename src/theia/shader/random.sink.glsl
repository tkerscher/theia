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

void main() {
    uint stream = baseStream + gl_GlobalInvocationID.x;
    uint count = gl_WorkGroupID.y * DRAWS * BATCH_SIZE + gl_LocalInvocationID.y;
    uint idx = gl_GlobalInvocationID.x * samples + count;

    for (
        uint i = 0;
        i < DRAWS && count < samples;
        ++i, idx += BATCH_SIZE, count += BATCH_SIZE
    ) {
        u[idx] = random(stream, count + baseCount);
    }
}
