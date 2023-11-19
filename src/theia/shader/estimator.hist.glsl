#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_float : require

//check if macro settings set
#ifndef BATCH_SIZE
#error "missing macro settings: BATCH_SIZE"
#endif
#ifndef N_BINS
#error "missing macro settings: N_BINS"
#endif
#ifndef N_PHOTONS
#error "missing macro settings: N_PHOTONS"
#endif

layout(local_size_x = BATCH_SIZE, local_size_y = N_PHOTONS) in;

#include "hits.glsl"
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

layout(scalar) readonly buffer HitQueueBuffer {
    uint hitCount;
    HitQueue hitQueue;
};

layout(scalar) uniform Parameters {
    float t0;
    float binSize;
    uint detectorId;
} params;

void main() {
    uint hitIdx = gl_GlobalInvocationID.x;
    uint phIdx = gl_GlobalInvocationID.y;
    uint idx = gl_LocalInvocationID.x;
    uint stride = gl_WorkGroupSize.x * gl_WorkGroupSize.y;

    //clear localHist
    for (uint i = idx; i < N_BINS; i += stride)
        localHist[i] = 0.0;
    memoryBarrierShared();
    barrier();

    //we can't use early returns if we want to use barrier()
    //-> if guard as range check
    if (hitIdx < hitCount) {
        //load hit
        vec3 pos = vec3(
            hitQueue.posX[hitIdx],
            hitQueue.posY[hitIdx],
            hitQueue.posZ[hitIdx]);
        vec3 dir = vec3(
            hitQueue.dirX[hitIdx],
            hitQueue.dirY[hitIdx],
            hitQueue.dirZ[hitIdx]);
        vec3 nrm = vec3(
            hitQueue.nrmX[hitIdx],
            hitQueue.nrmY[hitIdx],
            hitQueue.nrmZ[hitIdx]);
        //load photon
        float wavelength = hitQueue.wavelength[phIdx][hitIdx];
        float t_hit = hitQueue.time[phIdx][hitIdx];
        float contribution = hitQueue.contribution[phIdx][hitIdx];

        //calculate response
        float importance = response(pos, dir, nrm,
            wavelength, params.detectorId);
        float value = importance * contribution;
        
        //calculate affected bin
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
