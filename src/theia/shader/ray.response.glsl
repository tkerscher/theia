#ifndef _INCLUDE_RAY_RESPONSE
#define _INCLUDE_RAY_RESPONSE

#include "ray.glsl"
#include "response.common.glsl"

//util functions for creating HitItem from rays to declutter code

#ifdef POLARIZATION

/**
 * Creates a HitItem based on the given ray's current state using the hit
 * position and normal in object space.
*/
HitItem createHit(
    const PolarizedForwardRay ray,  ///< Ray generating a hit
    vec3 objHitPos,                 ///< Hit position in object space
    vec3 objHitNormal               ///< Hit normal in object space
) {
    float contrib = ray.state.lin_contrib * exp(ray.state.log_contrib);
    //normalize stokes
    contrib *= ray.stokes.x;
    vec4 stokes = ray.stokes / ray.stokes.x;

    //create HitItem
    return HitItem(
        objHitPos,
        ray.state.direction,
        objHitNormal,
        stokes,
        ray.polRef,
        ray.state.wavelength,
        ray.state.time,
        contrib
    );
}

#else

HitItem createHit(
    const UnpolarizedForwardRay ray,
    vec3 objHitPos,
    vec3 objHitNormal
) {
    float contrib = ray.state.lin_contrib * exp(ray.state.log_contrib);
    return HitItem(
        objHitPos,
        ray.state.direction,
        objHitNormal,
        ray.state.wavelength,
        ray.state.time,
        contrib
    );
}

#endif

#endif
