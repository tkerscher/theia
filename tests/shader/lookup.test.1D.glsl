#include "lookup.glsl"

layout(local_size_x = 32) in;

writeonly buffer OutputBuffer { float outValues[]; };

layout(scalar, push_constant) uniform PushConstant {
    Table1D table;
    float normalization;
} push;

void main() {
    uint i = gl_GlobalInvocationID.x;
    float u = float(i) / push.normalization;

    outValues[i] = lookUp(push.table, u);
}
