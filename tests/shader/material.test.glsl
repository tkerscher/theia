layout(local_size_x = 64) in;

//N = 32 * 64
#define N 2048
#define SEED 0xC0FFEEu

#include "material.glsl"
#include "random/util.glsl"
#include "util/hash.glsl"

struct Queue {
    float n[N];
    float vg[N];
    float mu_s[N];
    float log_phase[N];
    float angle[N];
};

writeonly buffer Result {
    float x[N];
    Queue inside;
    Queue outside;
};

writeonly buffer Flags{
    uint32_t flags[];
};

void main() {
    //use hash function to get pseudo random number
    uint i = gl_GlobalInvocationID.x;
    float u = normalizeUint(hash(i, SEED));
    x[i] = u;

    float lam = u * 300.0 + 400.0;  //[400.0, 700.0]
    float ct = 2.0 * u - 1.0;       //[ -1.0,   0.0]

    //fetch media
    const uint matIdx = 1;
    uint medInsideIdx, flagsInwards, medOutsideIdx, flagsOutwards;
    queryMaterialSide(matIdx, true, medInsideIdx, flagsInwards);
    queryMaterialSide(matIdx, false, medOutsideIdx, flagsOutwards);

    //query inside medium
    inside.n[i]         = lookUpMediaTable1D(REFRACTIVE_INDEX,   medInsideIdx, lam, -10.0);
    inside.vg[i]        = lookUpMediaTable1D(GROUP_VELOCITY,     medInsideIdx, lam, -10.0);
    inside.mu_s[i]      = lookUpMediaTable1D(SCATTERING_COEF,    medInsideIdx, lam, -10.0);
    inside.log_phase[i] = lookUpMediaTable1D(LOG_PHASE_FUNCTION, medInsideIdx,  ct, -10.0);
    inside.angle[i]     = lookUpMediaTable1D(PHASE_SAMPLING,     medInsideIdx,   u, -10.0);
    //query outside medium
    outside.n[i]         = lookUpMediaTable1D(REFRACTIVE_INDEX,   medOutsideIdx, lam, -10.0);
    outside.vg[i]        = lookUpMediaTable1D(GROUP_VELOCITY,     medOutsideIdx, lam, -10.0);
    outside.mu_s[i]      = lookUpMediaTable1D(SCATTERING_COEF,    medOutsideIdx, lam, -10.0);
    outside.log_phase[i] = lookUpMediaTable1D(LOG_PHASE_FUNCTION, medOutsideIdx,  ct, -10.0);
    outside.angle[i]     = lookUpMediaTable1D(PHASE_SAMPLING,     medOutsideIdx,   u, -10.0);

    //storing flags once is enough
    if (i == 0) {
        flags[0] = flagsInwards;
        flags[1] = flagsOutwards;
    }
}
