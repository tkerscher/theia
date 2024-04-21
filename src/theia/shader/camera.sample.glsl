#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif

layout(local_size_x = BATCH_SIZE) in;

#include "camera.queue.glsl"
//test for rare edge case:
//combine HostCameraRaySource & CameraRaySampler, but mismatch polarization
//(would require two different version of CameraQueue)
#ifdef CAMERA_QUEUE_POLARIZED
#define SAMPLER_CAMERA_QUEUE_POLARIZED
#undef LIGHT_QUEUE_POLARIZED
#endif
//user provided code
#include "rng.glsl"
#include "camera.glsl"

//check for queue mismatch
#ifdef _INCLUDE_CAMERARAYSOURCE_HOST
#if defined(SAMPLER_CAMERA_QUEUE_POLARIZED) != defined(CAMERA_QUEUE_POLARIZED)
#error "polarization mismatch in camera queue definition"
#endif
#endif

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
