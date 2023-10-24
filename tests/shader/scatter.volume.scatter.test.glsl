#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "scatter.volume.glsl"

layout(local_size_x = 32) in;

layout(scalar) readonly buffer Input{
    Ray rays[];
};
layout(scalar) readonly buffer RNG{
    float u[];
};

struct Result{
    Ray ray;
    float prob;
};
layout(scalar) writeonly buffer Output{
    Result results[];
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    Ray ray = rays[i];
    vec2 rng = vec2(u[2*i], u[2*i + 1]);
    
    float prob;
    scatter(ray, rng, prob);

    results[i] = Result(ray, prob);
}
