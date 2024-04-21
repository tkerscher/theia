#ifndef _INCLUDE_CAMERARAYSOURCE_COMMON
#define _INCLUDE_CAMERARAYSOURCE_COMMON

struct CameraRay {
    vec3 position;
    vec3 direction;
    float contrib;
    float timeDelta;

#ifdef POLARIZATION
    //reference frame of polarization indicating vertical linear polarization
    vec3 polRef;
#endif

    vec3 hitPosition;
    vec3 hitDirection;
    vec3 hitNormal;
};

#endif
