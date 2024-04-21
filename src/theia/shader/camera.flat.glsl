#ifndef _INCLUDE_CAMERARAYSOURCE_FLAT
#define _INCLUDE_CAMERARAYSOURCE_FLAT

#include "math.glsl"

layout(scalar) uniform CameraRayParams {
    float width;
    float height; //length
    float contrib;
    vec3 offset;
    mat3 trafo;
} cameraRayParams;

CameraRay sampleCameraRay(uint idx, uint dim) {
    //sample direction
    vec2 u = random2D(idx, dim);
    float cos_theta = u.x; //limit to upper hemisphere
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    float phi = TWO_PI * u.y;
    vec3 localDir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    vec3 rayDir = cameraRayParams.trafo * localDir;
    //flip local dir as it should hit the detector
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
    polRef = cameraRayParams.trafo * polRef;
#endif

    //sample position on detector
    u = random2D(idx, dim + 2);
    float localX = cameraRayParams.width * (u.x - 0.5);
    float localY = cameraRayParams.height * (u.y - 0.5);
    vec3 localPos = vec3(localX, localY, 0.0);
    //transform to sceen coord space
    vec3 rayPos = cameraRayParams.trafo * localPos + cameraRayParams.offset;

    //assemble camera ray
    return CameraRay(
        rayPos, rayDir,                 // ray pos / dir
        cameraRayParams.contrib, 0.0,   // contrib / time delta
#ifdef POLARIZATION
        polRef,
#endif
        localPos, localDir,             // hit pos / dir
        vec3(0.0, 0.0, 1.0)             // hit normal
    );
}

#endif
