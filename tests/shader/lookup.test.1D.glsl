#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "lookup.glsl"

layout(local_size_x = 32) in;

writeonly buffer OutputBuffer { float outValues[]; };

layout(push_constant) uniform PushConstant {
    Table1D table;
    float normalization;
} push;

void main() {
    uint i = gl_GlobalInvocationID.x;
    float u = float(i) / push.normalization;

    outValues[i] = lookUp(push.table, u);
}
