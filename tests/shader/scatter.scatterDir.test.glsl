#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "scatter.glsl"

layout(local_size_x = 32) in;

struct Query {
    vec3 inDir;
    float cos_theta;
    float phi;
};

layout(scalar) readonly buffer QueryBuffer{ Query q[]; };
layout(scalar) writeonly buffer ResultBuffer{ vec3 r[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    r[i] = scatterDir(q[i].inDir, q[i].cos_theta, q[i].phi);
}
