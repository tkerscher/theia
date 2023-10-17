#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

#include "sphere.glsl"

layout(local_size_x = 32) in;

struct Query{
    vec3 observer;
    vec3 direction;
};

layout(scalar) readonly buffer Input{
    Query queries[];
};
layout(scalar) writeonly buffer Output{
    float prob[];
};

layout(push_constant, scalar) uniform Push{
    Sphere sphere;
} push;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    Query q = queries[idx];
    prob[idx] = sampleSphereProb(push.sphere, q.observer, q.direction);
}
