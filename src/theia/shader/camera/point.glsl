#ifndef _INCLUDE_CAMERARAYSOURCE_POINT
#define _INCLUDE_CAMERARAYSOURCE_POINT

#include "math.glsl"
#include "util.sample.glsl"

uniform CameraParams {
    vec3 position;
    float timeDelta;
} cameraParams;

CameraRay sampleCameraRay(float wavelength, uint idx, inout uint dim) {
    //sample direction
    vec3 dir = sampleUnitSphere(random2D(idx, dim));
    vec3 polRef = perpendicularTo(dir);

    //assemble camera ray
    return createCameraRay(
        cameraParams.position,  //ray position
        dir,                    //ray direction
        polRef,                 //ray polRef
        mat4(1.0),              //ray mueller matrix
        FOUR_PI,                //contrib
        cameraParams.timeDelta, //time delta
        vec3(0.0, 0.0, 0.0),    //hit position
        -dir,                   //hit direction
        dir,                    //hit normal
        polRef                  //hit polRef
    );
}

#endif
