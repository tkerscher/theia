#ifndef _INCLUDE_CAMERARAYSOURCE_POINT
#define _INCLUDE_CAMERARAYSOURCE_POINT

#include "math.glsl"

layout(scalar) uniform CameraRayParams {
    vec3 position;
    float timeDelta;
} cameraRayParams;

CameraRay sampleCameraRay(float wavelength, uint idx, inout uint dim) {
    //sample direction
    vec2 u = random2D(idx, dim);
    float phi = TWO_PI * u.x;
    float cos_theta = 2.0 * u.y - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    vec3 dir = vec3(
        sin_theta * sin(phi),
        sin_theta * cos(phi),
        cos_theta
    );
    vec3 polRef = perpendicularTo(dir);

    //assemble camera ray
    return createCameraRay(
        cameraRayParams.position,   //ray position
        dir,                        //ray direction
        polRef,                     //ray polRef
        mat4(1.0),                  //ray mueller matrix
        FOUR_PI,                    //contrib
        cameraRayParams.timeDelta,  //time delta
        vec3(0.0, 0.0, 0.0),        //hit position
        -dir,                       //hit direction
        dir,                        //hit normal
        polRef                      //hit polRef
    );
}

#endif
