layout(local_size_x = 32) in;

#include "material.glsl"

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
    float m12;
    float m22;
    float m33;
    float m34;
};

// TODO: Figure out how I can apply readonly here...
layout(scalar) buffer QueryBuffer{ Query queries[]; };
layout(scalar) writeonly buffer Results{ Result results[]; };
layout(scalar) writeonly buffer Flags{ uint32_t flags[]; };

Result sampleMedium(const Medium medium, float lambda, float theta, float eta) {
    MediumConstants constants = lookUpMedium(medium, lambda);
    bool hasMedium = uint64_t(medium) != 0;
    float t = 0.5 * (theta + 1.0); //remap [-1,1] -> [0,1]
    return Result(
        constants.n,
        constants.vg,
        constants.mu_s,
        constants.mu_e,
        hasMedium ? lookUp(medium.log_phase, t) : 0.0,
        hasMedium ? lookUp(medium.phase_sampling, eta) : 0.0,
        hasMedium ? lookUp(medium.phase_m12, t) : 0.0,
        hasMedium ? lookUp(medium.phase_m22, t) : 0.0,
        hasMedium ? lookUp(medium.phase_m33, t) : 0.0,
        hasMedium ? lookUp(medium.phase_m34, t) : 0.0
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
        flags[0] = mat.flagsInwards;
        flags[1] = mat.flagsOutwards;
    }
}
