#ifndef _INCLUDE_RESPONSE_COMMON
#define _INCLUDE_RESPONSE_COMMON

struct HitItem {
    vec3 position;
    vec3 direction;
    vec3 normal;

#ifdef POLARIZATION
    vec4 stokes; //(normalized) stokes vector
    vec3 polRef; //orientation of reference frame
#endif
    
    float wavelength;
    float time;
    float contrib;

    int objectId; //Id of the intersected geometry
};

#endif
