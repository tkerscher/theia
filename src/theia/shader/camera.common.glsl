#ifndef _INCLUDE_CAMERA_COMMON
#define _INCLUDE_CAMERA_COMMON

#ifdef POLARIZATION
#include "math.glsl"
#endif

struct CameraSample {
    vec3 position;
    vec3 normal;

    float contrib;
    
    int objectId;

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

    int objectId;
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
    int objectId,
    vec3 hitPosition,
    vec3 hitNormal
) {
    return CameraSample(
        position,
        normal,
        contrib,
        objectId,
        hitPosition,
        hitNormal
    );
}

CameraSample createCameraSample(
    vec3 position,
    vec3 normal,
    float contrib,
    vec3 hitPosition,
    vec3 hitNormal
) {
    return createCameraSample(position, normal, contrib, -1, hitPosition, hitNormal);
}

CameraSample createCameraSample(
    vec3 position,
    vec3 normal,
    float contrib,
    int objectId
) {
    return CameraSample(
        position,
        normal,
        contrib,
        objectId,
        vec3(0.0),
        vec3(0.0)
    );
}

CameraSample createCameraSample(
    vec3 position,
    vec3 normal,
    float contrib
) {
    return createCameraSample(position, normal, contrib, -1);
}

CameraRay createCameraRay(
    vec3 rayPosition,
    vec3 rayDirection,
    vec3 rayPolRef,
    mat4 mueller,
    float contrib,
    float timeDelta,
    int objectId,
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
            hitNormal,
            objectId
        )
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
    return createCameraRay(
        rayPosition,
        rayDirection,
        rayPolRef,
        mueller,
        contrib,
        timeDelta,
        -1,
        hitPosition,
        hitDirection,
        hitNormal,
        hitPolRef
    );
}

CameraRay createCameraRay(
    vec3 rayPosition,
    vec3 rayDirection,
    float contrib,
    float timeDelta,
    int objectId,
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
            hitNormal,
            objectId
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
    return createCameraRay(
        rayPosition,
        rayDirection,
        contrib,
        timeDelta,
        -1,
        hitPosition,
        hitDirection,
        hitNormal
    );
}

#endif
