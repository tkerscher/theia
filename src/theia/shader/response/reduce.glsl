layout(local_size_x = 128) in;

#include "util/buffers.glsl"

#ifdef USE_DOUBLE
#define Buffer DoubleBuffer
#define Scalar float64_t
#define ZERO 0.0LF
#else
#define Scalar float32_t
#define Buffer FloatBuffer
#define ZERO 0.0
#endif

uniform Params {
    uvec2 bufferInAdr;
    uvec2 bufferOutAdr;

    uint binCount;
    uint bufferCount;

    Scalar norm;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= params.binCount) return;

    //add same bin accross all input buffers and save in output buffer
    //use compensated Kahan summation to reduce error
    //See https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    precise Scalar sum = ZERO;
    precise Scalar c = ZERO;
    Buffer binsIn = Buffer(params.bufferInAdr);
    for (uint n = 0; n < params.bufferCount; ++n, i += params.binCount) {
        precise Scalar next = binsIn.values[i];
        precise Scalar t = sum + next;
        if (abs(sum) >= abs(next)) {
            c += (sum - t) + next;
        }
        else {
            c += (next - t) + sum;
        }
        sum = t;
    }
    sum = sum + c;

    Buffer binsOut = Buffer(params.bufferOutAdr);
    binsOut.values[gl_GlobalInvocationID.x] = sum * params.norm;
}
