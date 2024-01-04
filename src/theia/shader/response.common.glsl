#ifndef _INCLUDE_RESPONSE_COMMON
#define _INCLUDE_RESPONSE_COMMON

struct HitItem {
    vec3 position;
    vec3 direction;
    vec3 normal;
    
    float wavelength;
    float time;
    float contrib;
};

#endif
