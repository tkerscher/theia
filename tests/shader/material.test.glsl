layout(local_size_x = 32) in;

#include "material.glsl"

struct Query {
    uint materialIdx;
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
buffer QueryBuffer{ Query queries[]; };
writeonly buffer Results{ Result results[]; };
writeonly buffer Flags{ uint32_t flags[]; };

Result sampleMedium(const uint mediumIdx, float lambda, float theta, float eta) {
    MediumConstants constants = lookUpMedium(mediumIdx, lambda);
    float t = 0.5 * (theta + 1.0); //remap [-1,1] -> [0,1]
    Table1D log_phase = Table1D(uint64_t(0));
    Table1D phase_sampling = Table1D(uint64_t(0));
    Table1D phase_m12 = Table1D(uint64_t(0));
    Table1D phase_m22 = Table1D(uint64_t(0));
    Table1D phase_m33 = Table1D(uint64_t(0));
    Table1D phase_m34 = Table1D(uint64_t(0));
    if (!isVacuum(mediumIdx)) {
        log_phase = loadMediaSlot_Table1D(LOG_PHASE_FUNCTION, mediumIdx);
        phase_sampling = loadMediaSlot_Table1D(PHASE_SAMPLING, mediumIdx);
        phase_m12 = loadMediaSlot_Table1D(PHASE_M12, mediumIdx);
        phase_m22 = loadMediaSlot_Table1D(PHASE_M22, mediumIdx);
        phase_m33 = loadMediaSlot_Table1D(PHASE_M33, mediumIdx);
        phase_m34 = loadMediaSlot_Table1D(PHASE_M34, mediumIdx);
    }
    return Result(
        constants.n,
        constants.vg,
        constants.mu_s,
        constants.mu_e,
        lookUp(log_phase, t),
        lookUp(phase_sampling, eta),
        lookUp(phase_m12, t),
        lookUp(phase_m22, t),
        lookUp(phase_m33, t),
        lookUp(phase_m34, t)
    );
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    float wavelength = queries[i].wavelength;
    float theta = queries[i].theta;
    float eta = queries[i].eta;
    uint matIdx = queries[i].materialIdx;

    uint medInsideIdx, flagsInwards, medOutsideIdx, flagsOutwards;
    queryMaterialSide(matIdx, true, medInsideIdx, flagsInwards);
    queryMaterialSide(matIdx, false, medOutsideIdx, flagsOutwards);

    // sample both inside and outside medium
    results[2*i + 0] = sampleMedium(medInsideIdx, wavelength, theta, eta);
    results[2*i + 1] = sampleMedium(medOutsideIdx, wavelength, theta, eta);

    //storing flags once is enough
    if (i == 0) {
        flags[0] = flagsInwards;
        flags[1] = flagsOutwards;
    }
}
