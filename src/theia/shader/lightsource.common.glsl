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
    
    float wavelength;
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
    float wavelength,
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
        wavelength,
        startTime,
        contrib
    );
#else
    return SourceRay(
        position,
        direction,
        wavelength,
        startTime,
        contrib
    );
#endif
}
SourceRay createSourceRay(
    vec3 position,
    vec3 direction,
    float startTime,
    WavelengthSample photon
) {
    return createSourceRay(
        position,
        direction,
        photon.wavelength,
        startTime,
        photon.contrib
    );
}

//create polarized source ray
SourceRay createSourceRay(
    vec3 position,
    vec3 direction,
    vec4 stokes,
    vec3 polRef,
    float wavelength,
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
        wavelength,
        startTime,
        contrib
    );
}
SourceRay createSourceRay(
    vec3 position,
    vec3 direction,
    float startTime,
    vec4 stokes,
    vec3 polRef,
    WavelengthSample photon
) {
    return createSourceRay(
        position,
        direction,
        stokes,
        polRef,
        photon.wavelength,
        startTime,
        photon.contrib
    );
}

#endif
