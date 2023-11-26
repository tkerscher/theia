#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_vote : require

#include "sphere.intersect.glsl"

layout(local_size_x = 32) in;

struct Query{
    vec3 pos;
    vec3 dir;
};

layout(scalar) readonly buffer Queries {
    Query queries[];
};
layout(scalar) writeonly buffer Result {
    float t[];
};

layout(push_constant, scalar) uniform Push {
    vec3 pos;
    float r;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    Query q = queries[idx];
    t[idx] = intersectSphere(Sphere(pos, r), q.pos, q.dir);
}
