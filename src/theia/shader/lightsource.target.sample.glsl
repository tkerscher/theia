layout(local_size_x = 32) in;

#include "math.glsl"
#include "wavelengthsource.common.glsl"

#include "rng.glsl"
#include "photon.glsl"
#include "target.glsl"

struct Result {
    float wavelength;
    vec3 position;
    vec3 normal;
    float contrib;
};
writeonly buffer ResultBuffer { Result r[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= BATCH_SIZE) return;
    uint dim = 0;

    //sample wavelength
    WavelengthSample photon = sampleWavelength(idx, dim);
    float lambda = photon.wavelength;
    //sample target
    vec3 samplePos, sampleNrm;
    float contrib = sampleLightTarget(lambda, samplePos, sampleNrm, idx, dim);

    //save result
    r[idx] = Result(lambda, samplePos, sampleNrm, contrib);
}
