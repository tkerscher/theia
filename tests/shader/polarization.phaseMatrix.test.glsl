#include "polarization.glsl"

layout(local_size_x = 32) in;

struct Query{
    vec4 stokes;
    float cos_theta;
};

layout(scalar) readonly buffer QueryBuffer{ Query q[]; };
layout(scalar) writeonly buffer ResultBuffer{ vec4 result[]; };

layout(push_constant) uniform Push {
    uvec2 medium;
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    mat4 phase = lookUpPhaseMatrix(Medium(medium), q[i].cos_theta);
    result[i] = phase * q[i].stokes;
}
