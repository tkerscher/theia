#ifndef _INCLUDE_RAY_RESPONSE
#define _INCLUDE_RAY_RESPONSE

#include "math.glsl"
#include "ray.glsl"
#include "response.common.glsl"

//util functions for creating HitItem from rays to declutter code

#ifdef POLARIZATION

#include "polarization.glsl"

/**
 * Creates a HitItem based on the given ray's current state using the hit
 * position and normal in object space.
*/
HitItem createHit(
    const PolarizedForwardRay ray,  ///< Ray generating a hit
    vec3 objHitPos,                 ///< Hit position in object space
    vec3 objHitNormal,              ///< Hit normal in object space
    int objectId,                   ///< Id of the hit object
    mat3 worldToObj                 ///< Transformation from world to object space
) {
    float contrib = ray.state.lin_contrib * exp(ray.state.log_contrib);

    //transform ray direction to object space
    vec3 objHitDir = normalize(worldToObj * ray.state.direction);

    //create pol ref in object space
    vec3 hitPolRef = perpendicularTo(objHitDir, objHitNormal);
    //for non-orthogonal transformations the transformed polRef may not lie in
    //the plane of incidence, but require a rotation of the stokes parameter
    //this also handles the case where the plane of incidence is degenerate
    //and the "random" chosen references frame do not align
    vec3 objPolRef = normalize(worldToObj * ray.polRef);
    mat4 mueller = alignPolRef(objHitDir, objPolRef, hitPolRef);
    vec4 stokes = mueller * ray.stokes;

    //normalize stokes
    contrib *= stokes.x;
    stokes = stokes / stokes.x;

    //create HitItem
    return HitItem(
        objHitPos,
        objHitDir,
        objHitNormal,
        stokes,
        hitPolRef,
        ray.state.wavelength,
        ray.state.time,
        contrib,
        objectId
    );
}

//createHit() with PolarizedBackwardRay not possible, as it gives us a Mueller
//matrix, whereas HitItem expects a Stokes vector

#else

HitItem createHit(
    const RayState state,
    vec3 objHitPos,
    vec3 objHitNormal,
    int objectId,
    mat3 worldToObj
) {
    float contrib = state.lin_contrib * exp(state.log_contrib);

    //transform ray direction to object space
    vec3 objHitDir = normalize(worldToObj * state.direction);

    return HitItem(
        objHitPos,
        objHitDir,
        objHitNormal,
        state.wavelength,
        state.time,
        contrib,
        objectId
    );
}

HitItem createHit(
    const UnpolarizedForwardRay ray,
    vec3 objHitPos,
    vec3 objHitNormal,
    int objectId,
    mat3 worldToObj
) {
    return createHit(ray.state, objHitPos, objHitNormal, objectId, worldToObj);
}

HitItem createHit(
    const UnpolarizedBackwardRay ray,
    vec3 objHitPos,
    vec3 objHitNormal,
    int objectId,
    mat3 worldToObj
) {
    return createHit(ray.state, objHitPos, objHitNormal, objectId, worldToObj);
}

#endif

#endif
