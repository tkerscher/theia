#ifndef _INCLUDE_CAMERA_POINT
#define _INCLUDE_CAMERA_POINT

#include "math.glsl"
#include "util/sample.glsl"

uniform CameraParams {
    vec3 position;
    float timeDelta;

    uint mediumIdx;
    int objectId;
} cameraParams;

BackwardRay sampleCameraRay(
    float wavelength,
    out CameraHit hit,
    uint idx, inout uint dim
) {
    vec3 dir = sampleUnitSphere(random2D(idx, dim));

    hit = createCameraHit(
        vec3(0.0),  //hit position
        -dir,       //hit direction
        dir,        //hit normal
        cameraParams.objectId
    );
    return createBackwardRay(
        cameraParams.position,
        dir,
        wavelength,
        cameraParams.mediumIdx,
        cameraParams.timeDelta,
        FOUR_PI
    );
}

#endif
