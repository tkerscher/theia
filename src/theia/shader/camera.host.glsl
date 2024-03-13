#ifndef _INCLUDE_CAMERARAYSOURCE_HOST
#define _INCLUDE_CAMERARAYSOURCE_HOST

#include "camera.queue.glsl"

layout(scalar) readonly buffer CameraQueueIn {
    CameraQueue queue;
} cameraQueueIn;

CameraRay sampleCameraRay(uint idx, uint dim) {
    LOAD_CAMERA(ray, cameraQueueIn.queue, idx)
    return ray;
}

#endif
