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
    float mu_s;
    float mu_e;
    float log_phase;
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
        consts.mu_s,
        consts.mu_e,
        lookUp(push.medium.log_phase, 0.5 * (theta + 1.0)),
        lookUp(push.medium.phase_sampling, eta)
    );
}
