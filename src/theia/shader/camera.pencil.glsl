#ifndef _INCLUDE_CAMERARAYSOURCE_PENCIL
#define _INCLUDE_CAMERARAYSOURCE_PENCIL

layout(scalar) uniform CameraRayParams {
    vec3 rayPosition;
    vec3 rayDirection;

    float timeDelta;

    vec3 hitPosition;
    vec3 hitDirection;
    vec3 hitNormal;
} cameraRayParams;

CameraRay sampleCameraRay(uint idx, uint dim) {
    return CameraRay(
        cameraRayParams.rayPosition,
        cameraRayParams.rayDirection,
        1.0, //contrib
        cameraRayParams.timeDelta,
        cameraRayParams.hitPosition,
        cameraRayParams.hitDirection,
        cameraRayParams.hitNormal
    );
}

#endif
