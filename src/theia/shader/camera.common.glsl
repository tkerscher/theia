#ifndef _INCLUDE_CAMERA_COMMON
#define _INCLUDE_CAMERA_COMMON

#ifdef POLARIZATION
#include "math.glsl"
#endif

struct CameraSample {
    vec3 position;
    vec3 normal;

    float contrib;

    //Can be used as cache in camera sampling
    //May not be used outside camera
    vec3 hitPosition;
    vec3 hitNormal;
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

CameraSample createCameraSample(
    vec3 position,
    vec3 normal,
    float contrib,
    vec3 hitPosition,
    vec3 hitNormal
) {
    return CameraSample(
        position,
        normal,
        contrib,
        hitPosition,
        hitNormal
    );
}

CameraSample createCameraSample(
    vec3 position,
    vec3 normal,
    float contrib
) {
    return CameraSample(
        position,
        normal,
        contrib,
        vec3(0.0),
        vec3(0.0)
    );
}

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
