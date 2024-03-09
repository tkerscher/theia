#extension GL_GOOGLE_include_directive :require
#extension GL_EXT_scalar_block_layout : require

#include "sphere.glsl"

layout(local_size_x = 32) in;

struct Result {
    vec3 dir;
    float prob;
};

layout(scalar) readonly buffer Input{
    vec3 observer[];
};
layout(scalar) writeonly buffer Output{
    Result result[];
};
layout(scalar) readonly buffer RNG{
    float u[];
};

layout(push_constant, scalar) uniform Push{
    Sphere sphere;
} push;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    vec2 rng = vec2(u[2*idx], u[2*idx + 1]);
    vec3 o = observer[idx];

    float p;
    vec3 dir = sampleSphere(push.sphere, o, rng, p);
    result[idx] = Result(dir, p);
}
