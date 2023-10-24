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
    float mu_s;
    float mu_e;
    float log_phase;
    float angle; //cos theta
};

// TODO: Figure out how I can apply readonly here...
layout(scalar) buffer QueryBuffer{ Query queries[]; };
layout(scalar) writeonly buffer Results{ Result results[]; };
layout(scalar) writeonly buffer Flags{ uint32_t flags[]; };

Result sampleMedium(const Medium medium, float lambda, float theta, float eta) {
    MediumConstants constants = lookUpMedium(medium, lambda);
    bool hasMedium = uint64_t(medium) != 0;
    return Result(
        constants.n,
        constants.vg,
        constants.mu_s,
        constants.mu_e,
        hasMedium ? lookUp(medium.log_phase, 0.5 * (theta + 1.0)) : 0.0,
        hasMedium ? lookUp(medium.phase_sampling, eta) : 0.0
    );
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    float wavelength = queries[i].wavelength;
    float theta = queries[i].theta;
    float eta = queries[i].eta;
    Material mat = queries[i].material;

    // sample both inside and outside medium
    results[2*i + 0] = sampleMedium(mat.inside, wavelength, theta, eta);
    results[2*i + 1] = sampleMedium(mat.outside, wavelength, theta, eta);

    //storing flags once is enough
    if (i == 0) {
        flags[0] = mat.flags;
    }
}
