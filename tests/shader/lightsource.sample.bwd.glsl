layout(local_size_x = 32) in;

#define LAM_MIN 400.0
#define LAM_MAX 800.0
#define DIM 100.0

#include "lightsource.common.glsl"

#include "rng.glsl"
#include "light.glsl"
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
    uint dim = 0;

    //sample wavelength
    float lambda = mix(LAM_MIN, LAM_MAX, random(i, dim));
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
