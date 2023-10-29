#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_float : require

//default settings; overwritten by preamble
#ifndef BATCH_SIZE
#define BATCH_SIZE 128
#endif
#ifndef N_BINS
#define N_BINS 256
#endif
#ifndef N_PHOTONS
#define N_PHOTONS 4
#endif

layout(local_size_x = BATCH_SIZE, local_size_y = N_PHOTONS) in;

#include "ray.glsl"
//user provided respones function
//float response(
//    vec3 rayPos, vec3 rayDir, vec3 detNormal,
//    float wavelength, float log_radiance, uint detectorId)
#include "response.glsl"

//local histogram
shared float localHist[N_BINS];
//global histograms
struct Histogram {
    float bin[N_BINS];
};
layout(scalar) writeonly buffer Histograms {
    Histogram hist[];
};

//response queue as buffer reference allows to pass pointer instead of binding
//a buffer -> public API is "stateless"
layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer ResponseQueue {
    uint count;
    RayHit hits[];
};

layout(push_constant, scalar) uniform Push {
    uvec2 responseQueue;
    float t0;
    float binSize;
    uint detectorId;
} params;

void main() {
    uint rayIdx = gl_GlobalInvocationID.x;
    uint phIdx = gl_GlobalInvocationID.y;
    uint idx = gl_LocalInvocationID.x;
    uint stride = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    //fetch queue
    ResponseQueue queue = ResponseQueue(params.responseQueue);

    //clear localHist
    for (uint i = idx; i < N_BINS; i += stride)
        localHist[i] = 0.0;
    memoryBarrierShared();
    barrier();

    //we can't use early returns if we want to use barrier()
    //-> if guard as range check
    if (rayIdx < queue.count) {
        //load item
        RayHit ray = queue.hits[rayIdx];
        PhotonHit photon = ray.hits[phIdx];

        //calculate response
        float value = response(
            ray.position,
            ray.direction,
            ray.normal,
            photon.wavelength,
            photon.log_radiance,
            params.detectorId);
        value *= photon.throughput;
        
        //calculate affected bin
        float t_hit = photon.travelTime;
        uint bin = int(floor((t_hit - params.t0) / params.binSize));
        
        //update histogram if bin is in range
        if (bin >= 0 && bin < N_BINS) {
            atomicAdd(localHist[bin], value);
        }
    }

    //ensure we're finished with the local histogram
    memoryBarrierShared();
    barrier();

    //copy local histogram from shared memory to global memory
    uint histId = gl_WorkGroupID.x;
    for (uint i = idx; i < N_BINS; i += stride)
        hist[histId].bin[i] = localHist[i];
}
