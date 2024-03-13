#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif

layout(local_size_x = BATCH_SIZE) in;

#include "camera.queue.glsl"
//user provided code
#include "rng.glsl"
#include "camera.glsl"

//output queue
layout(scalar) writeonly buffer CameraQueueOut {
    uint sampleCount;
    CameraQueue queue;
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
    
    //sample camera
    CameraRay ray = sampleCameraRay(idx + sampleParams.baseCount, 0);
    //save sample
    SAVE_CAMERA(ray, queue, idx)

    //save the item count exactly once
    if (idx == 0) {
        sampleCount = sampleParams.count;
    }
}
