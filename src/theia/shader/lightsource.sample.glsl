#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

//check macro settings
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif
#ifndef N_LAMBDA
#error "N_LAMBDA not defined"
#endif

layout(local_size_x = BATCH_SIZE) in;

#include "lightsource.queue.glsl"
//user provided source
#include "light.glsl"


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

