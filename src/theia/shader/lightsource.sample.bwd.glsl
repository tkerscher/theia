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

layout(scalar) uniform SamplerParams {
    vec3 observer;
    uvec2 medium;    
} samplerParams;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= BATCH_SIZE) return;
    uint dim = 0;

    //sample wavelength
    WavelengthSample photon = sampleWavelength(i, dim);
    float lambda = photon.wavelength;
    //sample observer position if not specified
    vec3 observer = samplerParams.observer;
    if (any(isnan(observer))) {
        observer = vec3(
            mix(-DIM, DIM, random(i, dim)),
            mix(-DIM, DIM, random(i, dim)),
            mix(-DIM, DIM, random(i, dim))
        );
    }
    //sample observer normal
    vec3 normal = sampleHemisphere(random2D(i, dim));

    //sample light
    Medium medium = Medium(samplerParams.medium);
    MediumConstants c = lookUpMedium(medium, lambda);
    SourceRay ray = sampleLight(observer, normal, lambda, c, i, dim);

    //save result
    r[i] = Result(observer, normal, lambda, ray);
}
