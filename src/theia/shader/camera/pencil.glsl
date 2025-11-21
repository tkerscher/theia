#ifndef _INCLUDE_CAMERARAYSOURCE_PENCIL
#define _INCLUDE_CAMERARAYSOURCE_PENCIL

uniform CameraParams {
    vec3 rayPosition;
    vec3 rayDirection;
    
    vec3 rayPolRef;

    float timeDelta;

    vec3 hitPosition;
    vec3 hitDirection;
    vec3 hitNormal;
    vec3 hitPolRef;
} cameraParams;

CameraRay sampleCameraRay(float wavelength, uint idx, uint dim) {
    return createCameraRay(
        cameraParams.rayPosition,
        cameraParams.rayDirection,
        cameraParams.rayPolRef,
        mat4(1.0),
        1.0,
        cameraParams.timeDelta,
        cameraParams.hitPosition,
        cameraParams.hitDirection,
        cameraParams.hitNormal,
        cameraParams.hitPolRef
    );
}

#endif
