//check macro settings
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif

layout(local_size_x = BATCH_SIZE) in;

#include "lightsource.queue.glsl"
#include "wavelengthsource.queue.glsl"
//test for rare edge case:
//combine HostLightSource & LightSampler, but mismatch polarization
//(would require two different version of LightSourceQueue)
#ifdef LIGHT_QUEUE_POLARIZED
#define SAMPLER_LIGHT_QUEUE_POLARIZED
#undef LIGHT_QUEUE_POLARIZED
#endif
//user provided source
#include "rng.glsl"
#include "light.glsl"
#include "photon.glsl"

//check for queue mismatch
#ifdef _INCLUDE_LIGHTSOURCE_HOST
#if defined(SAMPLER_LIGHT_QUEUE_POLARIZED) != defined(LIGHT_QUEUE_POLARIZED)
#error "mismatch in light queue definition"
#endif
#endif

//output queue
layout(scalar) writeonly buffer LightQueueOut {
    uint sampleCount;
    LightSourceQueue data;
} lightQueue;
layout(scalar) writeonly buffer PhotonQueueOut {
    uint sampleCount;
    WavelengthQueue data;
} photonQueue;

//sample params
layout(scalar) uniform SampleParams {
    uint count;
    uint baseCount;

    uvec2 medium;
} sampleParams;

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    //range check
    if (idx >= sampleParams.count)
        return;

    //sample light
    WavelengthSample photon = sampleWavelength(
        idx + sampleParams.baseCount, dim);
    Medium medium = Medium(sampleParams.medium);
    SourceRay ray = sampleLight(
        photon.wavelength,
        lookUpMedium(medium, photon.wavelength),
        idx + sampleParams.baseCount, dim);
    //save sample
    SAVE_SAMPLE(ray, lightQueue.data, idx)
    SAVE_PHOTON(photon, photonQueue.data, idx)

    //save the item count exactly once
    if (gl_GlobalInvocationID.x == 0) {
        lightQueue.sampleCount = sampleParams.count;
        photonQueue.sampleCount = sampleParams.count;
    }
}
