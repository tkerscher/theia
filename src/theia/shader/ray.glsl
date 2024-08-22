#ifndef _INCLUDE_RAY
#define _INCLUDE_RAY

#include "lightsource.common.glsl"
#include "camera.common.glsl"

#include "material.glsl"

/**
Since GLSL has no inheritance, we need to get a creative to work around. We want
different types of rays describing their behaviour, e.g. unpolarized/polarized
or which direction they propagate. This allows to pass information about the
behavious at compile time without the need to carry flags along making the code
also more readable.

We achieve this by having the "base class" as member no matter what.
Later on we can emulate templates/generics via macros.
*/

//Common data/state every ray needs
struct RayState {
    vec3 position;
    vec3 direction;

    #ifndef USE_GLOBAL_MEDIUM
    uvec2 medium; //buffer_reference
    #endif

    float wavelength;
    float time;
    float lin_contrib;
    float log_contrib;

    MediumConstants constants;
};

RayState createRayState(
    const SourceRay source,
    const Medium medium
) {
    return RayState(
        source.position,
        normalize(source.direction), //just to be safe
        
        #ifndef USE_GLOBAL_MEDIUM
        uvec2(medium),
        #endif

        source.wavelength,
        source.startTime,
        source.contrib,     //lin_contrib
        0.0,                //log_contrib
        lookUpMedium(medium, source.wavelength)
    );
}

RayState createRayState(
    const CameraRay cam,
    const Medium medium,
    float wavelength
) {
    return RayState(
        cam.position,
        cam.direction,

        #ifndef USE_GLOBAL_MEDIUM
        uvec2(medium),
        #endif

        wavelength,
        cam.timeDelta,
        cam.contrib,        //lin_contrib
        0.0,                //log_contrib
        lookUpMedium(medium, wavelength)
    );
}

//util function to declutter code
//(we expect a similar function to be defined in case of a global medium)
#ifndef USE_GLOBAL_MEDIUM
Medium getMedium(const RayState ray) {
    return Medium(ray.medium);
}
#endif

//Here we handle the switch between polarized and unpolarized rays
//giving other code a common Ray type, so they don't have to bother
//with polarization on their own.

#ifdef POLARIZATION

/*******************************************************************************
*                                POLARIZED RAY                                 *
*******************************************************************************/

//for polarized simulation the ray has different behaviour and data depending on
//direction of the propagation, i.e. starting from the light source (forward) or
//from the camera/detector (backward)

//the reference frame of the polarization is defined such that polRef is
//perpendicular to the ray propagation, pointing to E_y component of the
//electric field.

struct PolarizedForwardRay {
    //unpolarized part we are extending
    RayState state;

    //polarization state
    vec4 stokes;
    vec3 polRef;
};

struct PolarizedBackwardRay {
    //unpolarized part we are extending
    RayState state;

    //polarized state
    mat4 mueller;
    vec3 polRef;

    //There is a slight gotcha here: The polarization state is still defined as
    //viewed by a photon travelling along a sampled path to make the code a bit
    //simpler. (We can simply multiply the mueller matrix to a sampled stokes
    //vector from a light source after aligning polRef). This however means for
    //polarization we have to flip the ray direction. Otherwise we have the
    //wrong sign in last two elements of the stokes vector.
};

PolarizedForwardRay createRay(
    const SourceRay source,
    const Medium medium
 ) {
    return PolarizedForwardRay(
        createRayState(source, medium),
        source.stokes,
        source.polRef
    );
 }

 PolarizedBackwardRay createRay(
    const CameraRay cam,
    const Medium medium,
    float wavelength
 ) {
    return PolarizedBackwardRay(
        createRayState(cam, medium, wavelength),
        mat4(1.0), //default init
        cam.polRef
    );
 }


//"typedef"
#define ForwardRay PolarizedForwardRay
#define BackwardRay PolarizedBackwardRay

#else

/*******************************************************************************
*                               UNPOLARIZED RAY                                *
*******************************************************************************/

struct UnpolarizedForwardRay {
    RayState state;
};

struct UnpolarizedBackwardRay {
    RayState state;
};

UnpolarizedForwardRay createRay(
    const SourceRay source,
    const Medium medium
) {
    return UnpolarizedForwardRay(
        createRayState(source, medium)
    );
}

UnpolarizedBackwardRay createRay(
    const CameraRay cam,
    const Medium medium,
    float wavelength
) {
    return UnpolarizedBackwardRay(
        createRayState(cam, medium, wavelength)
    );
}

//"typedef"
#define ForwardRay UnpolarizedForwardRay
#define BackwardRay UnpolarizedBackwardRay

#endif

#endif
