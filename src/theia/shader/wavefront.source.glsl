#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "ray.glsl"

//default settings; overwritten by preamble
#ifndef LOCAL_SIZE
#define LOCAL_SIZE 32
#endif
#ifndef N_PHOTONS
#define N_PHOTONS 4
#endif
#ifndef RNG_SAMPLES
#define RNG_SAMPLES 0
#endif

layout(local_size_x = LOCAL_SIZE) in;

//Light description
struct SourcePhoton{
    float wavelength;
    float log_radiance;
    float startTime;
    float probability;
};
struct SourceRay{
    vec3 position;
    vec3 direction;
    SourcePhoton photons[N_PHOTONS];
};
//user provided source: defines function SourceRay sample()
#include "light.glsl"

//redefine ray query as buffer reference to avoid the need to bind it
layout(buffer_reference, scalar, buffer_reference_align=4) writeonly buffer RayQueue {
    uint count;
    Ray rays[];
};

layout(push_constant, scalar) uniform Push {
    uvec2 rayQueue; // 8 bytes
    uvec2 medium;   // 8 bytes
    uint count;     // 4 bytes
    uint rngStride; // 4 bytes
} params;           //24 bytes

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.count)
        return;
    //retrieve buffer references
    RayQueue queue = RayQueue(params.rayQueue);
    Medium medium = Medium(params.medium);
    //calc rng stride, i.e. samples per stream
    uint rngStride = params.rngStride + RNG_SAMPLES;

    //sample light
    SourceRay sourceRay = sampleLight();

    //create photons
    Photon photons[N_PHOTONS];
    for (int i = 0; i < N_PHOTONS; ++i) {
        photons[i] = createPhoton(
            medium,
            sourceRay.photons[i].wavelength,
            sourceRay.photons[i].startTime,
            sourceRay.photons[i].log_radiance,
            sourceRay.photons[i].probability
        );
    }
    //create ray and place it on the queue
    //we assume that we fill an empty queue
    // -> no need for atomic counting
    queue.rays[idx] = Ray(
        sourceRay.position,
        sourceRay.direction,
        idx * rngStride + RNG_SAMPLES,
        params.medium,
        photons
    );

    //save the item count exactly once
    if (idx == 0) {
        queue.count = params.count;
    }
}
