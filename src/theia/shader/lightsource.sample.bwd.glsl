layout(local_size_x = 32) in;

#include "lightsource.common.glsl"
#include "wavelengthsource.common.glsl"

#include "rng.glsl"
#include "light.glsl"
#include "photon.glsl"
#include "util.sample.glsl"

struct Result {
    vec3 observer;
    vec3 normal;
    float wavelength;
    SourceRay ray;
};
layout(scalar) writeonly buffer ResultBuffer { Result r[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= BATCH_SIZE) return;
    uint dim = 0;

    //sample wavelength
    WavelengthSample photon = sampleWavelength(i, dim);
    float lambda = photon.wavelength;
    //sample observer position
    vec3 observer = vec3(
        mix(-DIM, DIM, random(i, dim)),
        mix(-DIM, DIM, random(i, dim)),
        mix(-DIM, DIM, random(i, dim))
    );
    //sample observer normal
    vec3 normal = sampleHemisphere(random2D(i, dim));

    //sample light
    SourceRay ray = sampleLight(observer, normal, lambda, i, dim);

    //save result
    r[i] = Result(observer, normal, lambda, ray);
}
