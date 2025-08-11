#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif

layout(local_size_x = BATCH_SIZE) in;

#include "camera.queue.glsl"
#include "wavelengthsource.queue.glsl"
//test for rare edge case:
//combine HostCameraRaySource & CameraRaySampler, but mismatch polarization
//(would require two different version of CameraQueue)
#ifdef CAMERA_QUEUE_POLARIZED
#define SAMPLER_CAMERA_QUEUE_POLARIZED
#undef CAMERA_QUEUE_POLARIZED
#endif
//user provided code
#include "rng.glsl"
#include "camera.glsl"
#include "photon.glsl"

//check for queue mismatch
#ifdef _INCLUDE_CAMERARAYSOURCE_HOST
#if defined(SAMPLER_CAMERA_QUEUE_POLARIZED) != defined(CAMERA_QUEUE_POLARIZED)
#error "polarization mismatch in camera queue definition"
#endif
#endif

//output queue
writeonly buffer CameraQueueOut {
    uint sampleCount;
    CameraQueue data;
} camQueue;
writeonly buffer PhotonQueueOut {
    uint sampleCount;
    WavelengthQueue data;
} photonQueue;

//sample params
uniform SampleParams {
    uint count;
    uint baseCount;
} sampleParams;

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    //range check
    if (idx >= sampleParams.count)
        return;
    
    //sample camera
    WavelengthSample photon = sampleWavelength(
        idx + sampleParams.baseCount, dim);
    CameraRay ray = sampleCameraRay(
        photon.wavelength,
        idx + sampleParams.baseCount,
        dim);
    //save sample
    SAVE_CAMERA(ray, camQueue.data, idx)
    SAVE_PHOTON(photon, photonQueue.data, idx)

    //save the item count exactly once
    if (gl_GlobalInvocationID.x == 0) {
        camQueue.sampleCount = sampleParams.count;
        photonQueue.sampleCount = sampleParams.count;
    }
}
