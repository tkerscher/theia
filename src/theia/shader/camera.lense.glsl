#ifndef _INCLUDE_CAMERARAYSOURCE_LENSE
#define _INCLUDE_CAMERARAYSOURCE_LENSE

#include "cosy.glsl"
#include "math.glsl"

layout(scalar) uniform CameraRayParams {
    float width;
    float height; //length
    float focal;
    float radius;
    float contrib;
    vec3 offset;
    mat3 trafo;
} cameraRayParams;

CameraRay sampleCameraRay(uint idx, uint dim) {
    //sample position on detector
    vec2 u = random2D(idx, dim);
    float localX = cameraRayParams.width * (u.x - 0.5);
    float localY = cameraRayParams.height * (u.y - 0.5);
    vec3 localPos = vec3(localX, localY, 0.0);
    //transform to sceen coord space
    vec3 rayPos = cameraRayParams.trafo * localPos + cameraRayParams.offset;

    //sample position on lense (disk)
    float r = sqrt(random(idx, dim + 2)) * cameraRayParams.radius;
    float theta = TWO_PI * random(idx, dim + 3);
    vec3 lensePos = vec3(r*sin(theta), r*cos(theta), cameraRayParams.focal);
    //calculate ray direction
    vec3 hitDir = normalize(localPos - lensePos);
    vec3 rayDir = cameraRayParams.trafo * (-hitDir);

    //p = 1/A_det * 1/A_lense * cos(theta)/d^2
    //  = 1/A_det * 1/A_lense * focal/d^3
    //contrib = 1/p = params.contrib * d^3
    float d = distance(localPos, lensePos);
    float contrib = cameraRayParams.contrib * (d*d*d);

    //assemble camera ray
    return CameraRay(
        rayPos, rayDir,     // ray pos / dir
        contrib, 0.0,       // contrib / time delta
        localPos, hitDir,   // hit pos / dir
        vec3(0.0, 0.0, 1.0) // hit normal
    );
}

#endif
