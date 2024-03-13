#ifndef _INCLUDE_CAMERARAYSOURCE_CONE
#define _INCLUDE_CAMERARAYSOURCE_CONE

#include "cosy.glsl"
#include "math.glsl"

layout(scalar) uniform CameraRayParams {
    vec3 conePos;
    vec3 coneDir;
    float cosOpeningAngle;
} cameraRayParams;

CameraRay sampleCameraRay(uint idx, uint dim) {
    //sample cone
    vec2 u = random2D(idx, dim);
    float phi = TWO_PI * u.x;
    float cos_theta = 1.0 - cameraRayParams.cosOpeningAngle * u.y;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    //construct local ray dir
    vec3 localDir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    //convert to global space
    vec3 rayDir = createLocalCOSY(normalize(cameraRayParams.coneDir)) * localDir;
    //flip local dir as it should point towards the detector
    localDir *= -1.0;

    //assemble camera ray
    return CameraRay(
        cameraRayParams.conePos, rayDir,             // global ray
        TWO_PI * cameraRayParams.cosOpeningAngle,   // contrib
        0.0,                                        // delta time
        vec3(0.0,0.0,0.0),                          // local hit pos
        localDir,                                   // local hit dir
        vec3(0.0,0.0,1.0)                           // local normal
    );
}

#endif
