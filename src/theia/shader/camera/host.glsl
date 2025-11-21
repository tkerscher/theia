#ifndef _INCLUDE_CAMERARAYSOURCE_HOST
#define _INCLUDE_CAMERARAYSOURCE_HOST

#include "camera.queue.glsl"

readonly buffer CameraQueueIn {
    CameraQueue queue;
} cameraQueueIn;

CameraRay sampleCameraRay(float wavelength, uint idx, uint dim) {
    LOAD_CAMERA(ray, cameraQueueIn.queue, idx)
    return ray;
}

#endif
