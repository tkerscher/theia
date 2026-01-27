#ifndef _INCLUDE_CAMERA_PENCIL
#define _INCLUDE_CAMERA_PENCIL

uniform CameraParams {
    vec3 rayPosition;
    vec3 rayDirection;

    float timeDelta;

    uint mediumIdx;
    int objectId;

    vec3 hitPosition;
    vec3 hitDirection;
    vec3 hitNormal;
} cameraParams;

BackwardRay sampleCameraRay(
    float wavelength,
    out CameraHit hit,
    uint idx, inout uint dim
) {
    hit = createCameraHit(
        cameraParams.hitPosition,
        cameraParams.hitDirection,
        cameraParams.hitNormal,
        cameraParams.objectId
    );
    return createBackwardRay(
        cameraParams.rayPosition,
        cameraParams.rayDirection,
        wavelength,
        cameraParams.mediumIdx,
        cameraParams.timeDelta,
        1.0
    );
}

#endif
