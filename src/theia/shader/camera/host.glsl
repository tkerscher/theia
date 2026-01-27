#ifndef _INCLUDE_CAMERA_HOST
#define _INCLUDE_CAMERA_HOST

uniform CameraParams {
    uvec2 queueAdr;
    uint queueSize;
} cameraParams;

BackwardRay sampleCameraRay(
    float wavelength,
    out CameraHit hit,
    uint idx, inout uint dim
) {
    //also populates hit
    BackwardRay ray = loadBackwardRay(
        cameraParams.queueAdr, cameraParams.queueSize, idx, hit
    );

    #ifdef CAMERA_OVERRIDE_WAVELENGTH
    ray.wavelength = wavelength;
    #endif

    return ray;
}

#endif
