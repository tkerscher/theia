#ifndef _INCLUDE_CAMERARAYSOURCE_CONE
#define _INCLUDE_CAMERARAYSOURCE_CONE

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
    mat3 trafo = createLocalCOSY(normalize(cameraRayParams.coneDir));
    vec3 rayDir = trafo * localDir;
    //flip local dir as it should point towards the detector
    localDir *= -1.0;

#ifdef POLARIZATION
    //create polarization reference frame in plane of incidence (normal e_z)
    vec3 polRef = vec3(localDir.y, -localDir.x, 0.0);
    //degenerate case: localDir || e_z
    float len = length(polRef);
    if (len < 1e-5) {
        polRef = vec3(1.0, 0.0, 0.0);
    }
    else {
        //normalize
        polRef /= len;
    }
    polRef = trafo * polRef;
#endif

    //assemble camera ray
    return CameraRay(
        cameraRayParams.conePos, rayDir,             // global ray
        TWO_PI * cameraRayParams.cosOpeningAngle,   // contrib
        0.0,                                        // delta time
#ifdef POLARIZATION
        polRef,
#endif
        vec3(0.0,0.0,0.0),                          // local hit pos
        localDir,                                   // local hit dir
        vec3(0.0,0.0,1.0)                           // local normal
    );
}

#endif
