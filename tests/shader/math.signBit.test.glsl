#include "math.glsl"

layout(local_size_x = 32) in;

readonly buffer FloatIn{ float x[]; };
writeonly buffer FloatOut{ float y[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    y[i] = signBit(x[i]);
}
