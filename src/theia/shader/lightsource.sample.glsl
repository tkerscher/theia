#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

//check macro settings
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif

layout(local_size_x = BATCH_SIZE) in;

#include "lightsource.queue.glsl"
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

//check for queue mismatch
#ifdef _INCLUDE_LIGHTSOURCE_HOST
#if defined(SAMPLER_LIGHT_QUEUE_POLARIZED) != defined(LIGHT_QUEUE_POLARIZED)
#error "mismatch in light queue definition"
#endif
#endif

//output queue
layout(scalar) writeonly buffer LightQueueOut {
    uint sampleCount;
    LightSourceQueue queue;
};

//sample params
layout(scalar) uniform SampleParams {
    uint count;
    uint baseCount;
} sampleParams;

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= sampleParams.count)
        return;

    //sample light
    SourceRay ray = sampleLight(idx + sampleParams.baseCount);
    //save sample
    SAVE_SAMPLE(ray, queue, idx);

    //save the item count exactly once
    if (idx == 0) {
        sampleCount = sampleParams.count;
    }
}

