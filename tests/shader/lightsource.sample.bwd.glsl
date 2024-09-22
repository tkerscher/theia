#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x = 32) in;

#define LAM_MIN 400.0
#define LAM_MAX 800.0
#define DIM 100.0

#include "lightsource.common.glsl"

#include "rng.glsl"
#include "light.glsl"

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
    float cos_theta = random(i, dim);
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    float phi = TWO_PI * random(i, dim);
    vec3 normal = vec3(
        sin_theta * sin(phi),
        sin_theta * cos(phi),
        cos_theta
    );

    //sample light
    SourceRay ray = sampleLight(observer, normal, lambda, i, dim);

    //save result
    r[i] = Result(observer, normal, lambda, ray);
}
