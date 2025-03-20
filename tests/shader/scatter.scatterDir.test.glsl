#include "scatter.volume.glsl"

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
