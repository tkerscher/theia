#ifndef _INCLUDE_CAMERARAYSOURCE_PENCIL
#define _INCLUDE_CAMERARAYSOURCE_PENCIL

layout(scalar) uniform CameraRayParams {
    vec3 rayPosition;
    vec3 rayDirection;
    
    vec3 rayPolRef;

    float timeDelta;

    vec3 hitPosition;
    vec3 hitDirection;
    vec3 hitNormal;
    vec3 hitPolRef;
} cameraRayParams;

CameraRay sampleCameraRay(float wavelength, uint idx, uint dim) {
    return createCameraRay(
        cameraRayParams.rayPosition,
        cameraRayParams.rayDirection,
        cameraRayParams.rayPolRef,
        mat4(1.0),
        1.0,
        cameraRayParams.timeDelta,
        cameraRayParams.hitPosition,
        cameraRayParams.hitDirection,
        cameraRayParams.hitNormal,
        cameraRayParams.hitPolRef
    );
}

#endif
