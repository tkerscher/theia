#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

//ensure compilation failes if no rng is given and random() is called
#ifndef NO_RNG

#include "rng.glsl"

uint rng_samples = 0; //counter
float random() {
    rng_samples++;
    return random(gl_GlobalInvocationID.x, rng_samples-1);
}
vec2 random2D() {
    rng_samples += 2;
    return random2D(gl_GlobalInvocationID.x, rng_samples-2);
}

#else

#define rng_samples 0

#endif

#include "ray.glsl"

//default settings; overwritten by preamble
#ifndef LOCAL_SIZE
#define LOCAL_SIZE 32
#endif
#ifndef N_PHOTONS
#define N_PHOTONS 4
#endif

layout(local_size_x = LOCAL_SIZE) in;

//Light description
struct SourcePhoton{
    float wavelength;
    float startTime;
    float lin_contrib;
    float log_contrib;
};
struct SourceRay{
    vec3 position;
    vec3 direction;
    SourcePhoton photons[N_PHOTONS];
};
//user provided source: defines function SourceRay sample()
#include "light.glsl"

//ray queue
layout(scalar) writeonly buffer RayQueue {
    uint count;
    Ray rays[];
} queue;

//sample params
layout(scalar) uniform SampleParams {
    uvec2 medium;
    uint count;
} sampleParams;

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= sampleParams.count)
        return;
    //retrieve medium
    Medium medium = Medium(sampleParams.medium);

    //sample light
    SourceRay sourceRay = sampleLight();

    //create photons
    Photon photons[N_PHOTONS];
    for (int i = 0; i < N_PHOTONS; ++i) {
        photons[i] = createPhoton(
            medium,
            sourceRay.photons[i].wavelength,
            sourceRay.photons[i].startTime,
            sourceRay.photons[i].lin_contrib,
            sourceRay.photons[i].log_contrib
        );
    }
    //create ray and place it on the queue
    //we assume that we fill an empty queue
    // -> no need for atomic counting
    queue.rays[idx] = Ray(
        sourceRay.position,
        normalize(sourceRay.direction), //just to be safe
        idx, rng_samples,    // stream, count
        sampleParams.medium,
        photons
    );

    //save the item count exactly once
    if (idx == 0) {
        queue.count = sampleParams.count;
    }
}

