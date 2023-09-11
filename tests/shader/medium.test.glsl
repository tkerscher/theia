#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

#include "material.glsl"

layout(local_size_x = 32) in;

struct Query {
    float wavelength;
    float theta;        // scattering angle
    float eta;          //random number
};

struct Result {
    float n;
    float vg;
    float mu_a;
    float mu_s;
    float phase;
    float angle; //cos theta
};

// TODO: Figure out how I can apply readonly here...
layout(scalar) buffer QueryBuffer{ Query queries[]; };
layout(scalar) writeonly buffer Results{ Result results[]; };

layout(push_constant) uniform PushConstant {
    Medium medium;
} push;

void main() {
    uint i = gl_GlobalInvocationID.x;
    // fetch params
    float wavelength = queries[i].wavelength;
    float theta = queries[i].theta;
    float eta = queries[i].eta;
    // look up
    MediumConstants consts = lookUpMedium(push.medium, wavelength);
    // look up other two tables and build result
    results[i] = Result(
        consts.n,
        consts.vg,
        consts.mu_a,
        consts.mu_s,
        lookUp(push.medium.phase, 0.5 * (theta + 1.0)),
        lookUp(push.medium.phase_sampling, eta)
    );
}