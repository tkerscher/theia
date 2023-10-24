#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "scatter.volume.glsl"

layout(local_size_x = 32) in;

struct Query{
    Ray ray;
    vec3 dir;
};
layout(scalar) readonly buffer Input{
    Query queries[];
};

layout(scalar) writeonly buffer Output{
    float p[];
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    Query q = queries[i];
    p[i] = scatterProb(q.ray, q.dir);
}
