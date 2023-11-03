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

layout(local_size_x = LOCAL_SIZE) in;

//rng api
layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer RNGBuffer {
    float u[];
};
/*!!! NEED TO INIT IN MAIN !!!*/
RNGBuffer rngBuffer;
uint rngIdx;
float rand() {
    return rngBuffer.u[rngIdx++];
}

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
layout(buffer_reference, scalar, buffer_reference_align=4) writeonly buffer RayQueue {
    uint count;
    Ray rays[];
};

layout(push_constant, scalar) uniform Push {
    uvec2 queue;    // 8 bytes
    uvec2 rngBuffer;// 8 bytes
    uvec2 medium;   // 8 bytes
    uint count;     // 4 bytes
    uint rngStride; // 4 bytes
} params;           //32 bytes

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.count)
        return;
    //retrieve queue
    RayQueue queue = RayQueue(params.queue);
    //init rng
    rngBuffer = RNGBuffer(params.rngBuffer);
    uint rngStride = params.rngStride;
    rngIdx = idx * rngStride;
    //retrieve medium
    Medium medium = Medium(params.medium);

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
        sourceRay.direction,
        rngIdx,
        params.medium,
        photons
    );

    //save the item count exactly once
    if (idx == 0) {
        queue.count = params.count;
    }
}
