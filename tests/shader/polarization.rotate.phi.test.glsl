#include "polarization.glsl"

layout(local_size_x = 32) in;

struct Query{
    vec4 stokes;
    float phi;
};

layout(scalar) readonly buffer QueryBuffer{ Query q[]; };
layout(scalar) writeonly buffer ResultBuffer{ vec4 r[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    r[i] = rotatePolRef(q[i].phi) * q[i].stokes;
}
