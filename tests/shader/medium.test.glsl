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
buffer QueryBuffer{ Query queries[]; };
writeonly buffer Results{ Result results[]; };

layout(scalar, push_constant) uniform PushConstant {
    uint mediumIdx;
} push;

void main() {
    uint i = gl_GlobalInvocationID.x;
    // fetch params
    float wavelength = queries[i].wavelength;
    float theta = queries[i].theta;
    float eta = queries[i].eta;
    // look up
    MediumConstants consts = lookUpMedium(push.mediumIdx, wavelength);
    // look up other two tables and build result
    Table1D log_phase =  loadMediaSlot_Table1D(LOG_PHASE_FUNCTION, push.mediumIdx);
    Table1D phase_sampling = loadMediaSlot_Table1D(PHASE_SAMPLING, push.mediumIdx);
    results[i] = Result(
        consts.n,
        consts.vg,
        consts.mu_s,
        consts.mu_e,
        lookUp(log_phase, 0.5 * (theta + 1.0)),
        lookUp(phase_sampling, eta)
    );
}
