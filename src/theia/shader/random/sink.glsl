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

#if SAMPLE_DIM == 1
#define RNG_TYPE float
#define RNG_DRAW random_s
#elif SAMPLE_DIM == 2
#define RNG_TYPE vec2
#define RNG_DRAW random2D_s
#elif SAMPLE_DIM == 3
#define RNG_TYPE vec3
#define RNG_DRAW random3D_s
#elif SAMPLE_DIM == 4
#define RNG_TYPE vec4
#define RNG_DRAW random4D_s
#else
#error "Unsupported sample dimension!"
#endif

void main() {
    uint stream = baseStream + gl_GlobalInvocationID.x;
    uint count = gl_WorkGroupID.y * DRAWS * BATCH_SIZE + gl_LocalInvocationID.y;
    uint idx = gl_GlobalInvocationID.x * samples + count;

    for (
        uint i = 0;
        i < DRAWS && count < samples;
        ++i, idx += BATCH_SIZE, count += BATCH_SIZE
    ) {
        RNG_TYPE rand = RNG_DRAW(stream, count + baseCount);
        #if SAMPLE_DIM == 1
        u[idx] = rand;
        #else
        //TODO: We break coalesced memory access this way.
        //      Might want to fix that in the future
        for (uint d = 0; d < SAMPLE_DIM; ++d) {
            u[SAMPLE_DIM * idx + d] = rand[d];
        }
        #endif
    }
}
