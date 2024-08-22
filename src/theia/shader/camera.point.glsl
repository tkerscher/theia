#ifndef _INCLUDE_CAMERARAYSOURCE_POINT
#define _INCLUDE_CAMERARAYSOURCE_POINT

#include "math.glsl"

layout(scalar) uniform CameraRayParams {
    vec3 position;
    float timeDelta;
} cameraRayParams;

CameraRay sampleCameraRay(uint idx, uint dim) {
    //sample direction
    vec2 u = random2D(idx, dim); dim += 2;
    float phi = TWO_PI * u.x;
    float cos_theta = 2.0 * u.y - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    vec3 dir = vec3(
        sin_theta * sin(phi),
        sin_theta * cos(phi),
        cos_theta
    );

    //create polRef normal to ray direction
    #ifdef POLARIZATION
    vec3 polRef = createLocalCOSY(dir)[0];
    #endif

    //assemble camera ray
    return CameraRay(
        cameraRayParams.position,   //ray position
        dir,                        //ray direction
        FOUR_PI,                    //contrib
        cameraRayParams.timeDelta,  //ray ellapsed time
        #ifdef POLARIZATION
        polRef,                     //polRef (ignored if unpolarized)
        #endif
        vec3(0.0),                  //hit position
        -dir,                       //hit direction
        dir                         //hit normal
    );
}

#endif
