#ifndef _INCLUDE_CAMERA_COMMON
#define _INCLUDE_CAMERA_COMMON

#ifdef POLARIZATION
#include "math.glsl"
#endif

struct CameraSample {
    vec3 position;
    vec3 normal;

    float contrib;
};

struct CameraHit {
    #ifdef POLARIZATION
    vec3 polRef;
    #endif

    vec3 position;
    vec3 direction;
    vec3 normal;
};

struct CameraRay {
    vec3 position;
    vec3 direction;

    #ifdef POLARIZATION
    vec3 polRef;
    mat4 mueller;
    #endif

    float contrib;
    float timeDelta;

    CameraHit hit;
};

CameraRay createCameraRay(
    vec3 rayPosition,
    vec3 rayDirection,
    vec3 rayPolRef,
    mat4 mueller,
    float contrib,
    float timeDelta,
    vec3 hitPosition,
    vec3 hitDirection,
    vec3 hitNormal,
    vec3 hitPolRef
) {
    return CameraRay(
        rayPosition,
        rayDirection,
        #ifdef POLARIZATION
        rayPolRef,
        mueller,
        #endif
        contrib,
        timeDelta,
        CameraHit(
            #ifdef POLARIZATION
            hitPolRef,
            #endif
            hitPosition,
            hitDirection,
            hitNormal
        )
    );
}

CameraRay createCameraRay(
    vec3 rayPosition,
    vec3 rayDirection,
    float contrib,
    float timeDelta,
    vec3 hitPosition,
    vec3 hitDirection,
    vec3 hitNormal
) {
    return CameraRay(
        rayPosition,
        rayDirection,
        #ifdef POLARIZATION
        perpendicularTo(rayDirection), //any perpendicular will do
        mat4(
            //perfect depolarizer mueller matrix
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ),
        #endif
        contrib,
        timeDelta,
        CameraHit(
            #ifdef POLARIZATION
            perpendicularTo(hitDirection, hitNormal),
            #endif
            hitPosition,
            hitDirection,
            hitNormal
        )
    );
}

#endif
