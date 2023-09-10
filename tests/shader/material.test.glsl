#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

#include "material.glsl"

layout(local_size_x = 32) in;

struct Query {
    Material material; //!<-- 8 byte alignment !
    // Medium medium;
    float wavelength;
    float theta; // scattering angle
    float eta; //random number
    float padding; // padding for 8 byte alignment
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
layout(scalar) writeonly buffer Debug{ uint64_t d[]; };

Result sampleMedium(const Medium medium, float lambda, float theta, float eta) {
    MediumConstants constants = lookUpMedium(medium, lambda);
    bool hasMedium = uint64_t(medium) != 0;
    return Result(
        constants.n,
        constants.vg,
        constants.mu_a,
        constants.mu_s,
        hasMedium ? lookUp(medium.phase, 0.5 * (theta + 1.0)) : 0.0,
        hasMedium ? lookUp(medium.phase_sampling, eta) : 0.0
    );
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    float wavelength = queries[i].wavelength;
    float theta = queries[i].theta;
    float eta = queries[i].eta;
    Material mat = queries[i].material;
    //Medium med = queries[i].medium;
    //results[2*i + 0] = sampleMedium(med, wavelength, theta, eta);

    // if (i == 0) {
    //     d[0] = uint64_t(med);
    //     d[1] = uint64_t(med.n);
    //     d[2] = uint64_t(med.vg);
    //     d[3] = uint64_t(med.mu_a);
    //     d[4] = uint64_t(med.mu_s);
    //     d[5] = uint64_t(med.phase);
    //     d[6] = uint64_t(med.phase_sampling);
    // }

    // sample both inside and outside medium
    results[2*i + 0] = sampleMedium(mat.inside, wavelength, theta, eta);
    results[2*i + 1] = sampleMedium(mat.outside, wavelength, theta, eta);
}
