#include "rng.glsl"

layout(local_size_x = PARALLEL_STREAMS, local_size_y = BATCH_SIZE) in;

writeonly buffer RngSink {
    float u[];
};

uniform Params {
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
        #if SAMPLE_DIM == 1
        u[idx] = random_s(stream, count + baseCount);
        #elif SAMPLE_DIM == 2
        vec2 rand = random2D_s(stream, count + baseCount);
        //TODO: We break coalesced memory access this way.
        //      Might want to fix that in the future
        u[2*idx] = rand.x;
        u[2*idx + 1] = rand.y;
        #else
        #error "Unsupported sample dimension!"
        #endif
    }
}
