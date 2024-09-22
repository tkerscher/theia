#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "rng.glsl"
#include "scatter.volume.glsl"

layout(local_size_x = 32) in;

struct Query {
    vec3 dir;
    uvec2 medium;
};
layout(scalar) readonly buffer Input{
    Query queries[];
};

struct Result{
    vec3 dir;
    float prob;
};
layout(scalar) writeonly buffer Output{
    Result results[];
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    Query q = queries[i];
    
    float prob;
    vec3 dir = scatter(Medium(q.medium), q.dir, random2D_s(i, 0), prob);

    results[i] = Result(dir, prob);
}
