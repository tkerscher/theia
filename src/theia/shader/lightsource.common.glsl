#ifndef _INCLUDE_LIGHTSOURCE_SAMPLING
#define _INCLUDE_LIGHTSOURCE_SAMPLING

#include "wavelengthsource.common.glsl"

//only needed for polarization
#ifdef POLARIZATION
#include "math.glsl"
#endif

struct SourceRay {
    vec3 position;
    vec3 direction;

#ifdef POLARIZATION
    //stokes vector
    vec4 stokes;
    //polarization reference frame orientation
    //points to E_y, i.e. vertical polarization
    vec3 polRef;
#endif
    
    float startTime;
    float contrib;
};

//The following defines two helper function that handles the mismatch between
//sources creating (un)polarized samples and the environment enabling
//polarization, i.e. discards unwanted polarization or inits unpolarized light
//with correct reference frame

//create unpolarized source ray
SourceRay createSourceRay(
    vec3 position,
    vec3 direction,
    float startTime,
    float contrib
) {
#ifdef POLARIZATION
    //create unpolarized reference frame
    //createLocalCosy creates two perpendicular vectors to dir (first,second column)
    //-> choose any of them
    vec3 polRef = createLocalCOSY(direction)[0];
    return SourceRay(
        position,
        direction,
        vec4(1.0, 0.0, 0.0, 0.0), //stokes
        polRef,
        startTime,
        contrib
    );
#else
    return SourceRay(
        position,
        direction,
        startTime,
        contrib
    );
#endif
}

//create polarized source ray
SourceRay createSourceRay(
    vec3 position,
    vec3 direction,
    vec4 stokes,
    vec3 polRef,
    float startTime,
    float contrib
) {
    return SourceRay(
        position,
        direction,
#ifdef POLARIZATION
        stokes,
        polRef,
#endif
        startTime,
        contrib
    );
}

#endif
