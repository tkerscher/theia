#ifndef _INCLUDE_RAY_PROPAGATE
#define _INCLUDE_RAY_PROPAGATE

/*
Since GLSL does not allow to derive from other structs, we sidestep this problem
by encapsulating the "base struct" as a common member.
Functions on the base class are emulated by defining them on the base struct and
provide functions for the "derived struct" forwarding their member containing
the base struct to the implementation.

Not nice, but appears to be the smaller evil to me.
Plus, we bite the bullet here making code relying on it more readable.
*/

#include "ray.glsl"
#include "result.glsl"

struct PropagateParams {
    float scatterCoefficient;

    vec3 lowerBBoxCorner;
    vec3 upperBBoxCorner;
    float maxTime;

    float maxDist;
};

/**
 * Samples the distance until the next expected scatter event.
 * Handles non-scattering media by returning params.maxDist.
*/
float sampleScatterLength(
    const RayState ray,             ///< State of ray
    const PropagateParams params,   ///< Propagation params
    float u                         ///< Random number used for sampling
) {
    float dist = params.maxDist;
    //negative values indicate that we should IS scattering length
    float sampleCoef = params.scatterCoefficient <= 0.0 ?
        ray.constants.mu_s : params.scatterCoefficient;
    bool canScatter = ray.constants.mu_s > 0.0;
    if (canScatter) {
        //sample exponential distribution
        //use u -> 1.0 - u > 0.0 to be safe on the log
        dist = -log(1.0 - u) / sampleCoef;
    }
    return dist;
}
//specializations
float sampleScatterLength(
    const ForwardRay ray,
    const PropagateParams params,
    float u
) {
    return sampleScatterLength(ray.state, params, u);
}
float sampleScatterLength(
    const BackwardRay ray,
    const PropagateParams params,
    float u
) {
    return sampleScatterLength(ray.state, params, u);
}

/**
 * Updates ray state as if they traveled a given distance, i.e. without updating
 * the ray's position. Includes sampling propabilities and check for maxTime.
*/
ResultCode updateRay(
    inout RayState ray,             ///< State of ray to update
    const float dist,               ///< Distance travelled
    const PropagateParams params    ///< Propagation params
) {
    ray.log_contrib -= ray.constants.mu_e * dist;
    ray.time += dist / ray.constants.vg;

    //return result of boundary check
    return ray.time <= params.maxTime ? RESULT_CODE_SUCCESS : RESULT_CODE_RAY_DECAYED;
}
//specializations
ResultCode updateRay(
    inout ForwardRay ray,
    const float dist,
    const PropagateParams params
) {
    return updateRay(ray.state, dist, params);
}
ResultCode updateRay(
    inout BackwardRay ray,
    const float dist,
    const PropagateParams params
) {
    return updateRay(ray.state, dist, params);
}

/**
 * Updates ray with contribution from importance sampling.
 * Set hit to true, if the ray was stopped by a hit.
*/
void updateRayIS(
    inout RayState ray,             ///< Ray state
    float dist,                     ///< Distance travelled
    const PropagateParams params,   ///< Propagation params
    bool hit                        ///< Whether ray was stopped by a hit
) {
    //we only used IS, if scattering is possible
    //otherwise we used a constant maxDistance
    bool canScatter = ray.constants.mu_s > 0.0;
    if (!canScatter)
        return;
    
    //negative values indicate that we should IS scattering length
    float sampleCoef = params.scatterCoefficient <= 0.0 ?
        ray.constants.mu_s : params.scatterCoefficient;
    
    ray.log_contrib += sampleCoef * dist;
    if (!hit) {
        //if we hit anything, the actual prop is to travel at least dist
        // -> happens to cancel the coefficient
        // -> we need to divide by the scatter coef if we did not hit anything
        ray.lin_contrib /= sampleCoef;
    }
}
//specializations
void updateRayIS(
    inout ForwardRay ray,
    float dist,
    const PropagateParams params,
    bool hit
) {
    updateRayIS(ray.state, dist, params, hit);
}
void updateRayIS(
    inout BackwardRay ray,
    float dist,
    const PropagateParams params,
    bool hit
) {
    updateRayIS(ray.state, dist, params, hit);
}

/**
 * Propagates the ray in its current direction for the given distance.
 * Includes out-of-bounds checks for time and position.
*/
ResultCode propagateRay(
    inout RayState ray,           ///< State of ray to propagate
    float dist,                     ///< distance to propagate
    const PropagateParams params    ///< Propagation params
) {
    ray.position += dist * ray.direction;
    //boundary check
    bool outside =
        any(lessThan(ray.position, params.lowerBBoxCorner)) ||
        any(greaterThan(ray.position, params.upperBBoxCorner));
    
    ResultCode result = updateRay(ray, dist, params);
    return outside ? RESULT_CODE_RAY_LOST : result;
}
//specialization
ResultCode propagateRay(
    inout ForwardRay ray,
    float dist,
    const PropagateParams params
) {
    return propagateRay(ray.state, dist, params);
}
ResultCode propagateRay(
    inout BackwardRay ray,
    float dist,
    const PropagateParams params
) {
    return propagateRay(ray.state, dist, params);
}

//alignToHit
#ifdef POLARIZATION

#include "polarization.glsl"

/**
 * Aligns the given ray to the specified hit at its current position.
 * Ensures the polarization state to be in the plane of incidence (polRef is
 * perpendicular to it)
*/
void alignRayToHit(
    inout PolarizedForwardRay ray,
    vec3 hitNormal
) {
    //update polarization state: rotate to plane of incidence
    vec3 polRef;
    mat4 rotate = rotatePolRef(ray.state.direction, ray.polRef, hitNormal, polRef);
    ray.stokes = rotate * ray.stokes;
    ray.polRef = polRef;
}

void alignRayToHit(
    inout PolarizedBackwardRay ray,
    vec3 hitNormal
) {
    //update polarization state: rotate to plane of incidence
    vec3 polRef;
    mat4 rotate = rotatePolRef(ray.state.direction, ray.polRef, hitNormal, polRef);
    //remember that we treat the mueller matrix as seen by a photon, i.e. the
    //opposite direction. Thus the correct mueller matrix is the inverse or
    //the transpose as the rotation matrix is orthogonal.
    ray.mueller = ray.mueller * transpose(rotate);
    ray.polRef = polRef;
}

#else

void alignRayToHit(
    inout UnpolarizedForwardRay ray,
    vec3 hitNormal
) { /* Nothing to do */ }
void alignRayToHit(
    inout UnpolarizedBackwardRay ray,
    vec3 hitNormal
) { /* Nothing to do */ }

#endif

/**
 * Propagates the ray to the specified hit position and updates it accordingly.
 * Updates the polarization state to be in the plance of incidence (polRef is
 * perpendicular to it)
 * NOTE: This function does not check if hitPos lies in the ray's direction.
*/
ResultCode propagateRayToHit(
    inout RayState ray,             ///< Ray to propagate
    vec3 hitPos,                    ///< Hit position
    vec3 hitNormal,                 ///< Normal of the hit surface
    const PropagateParams params    ///< Propagation params
) {
    float dist = distance(ray.position, hitPos);
    ray.position = hitPos;
    return updateRay(ray, dist, params);
}
//Specializations
ResultCode propagateRayToHit(
    inout ForwardRay ray,
    vec3 hitPos,
    vec3 hitNormal,
    const PropagateParams params
) {
    alignRayToHit(ray, hitNormal);
    return propagateRayToHit(ray.state, hitPos, hitNormal, params);
}
ResultCode propagateRayToHit(
    inout BackwardRay ray,
    vec3 hitPos,
    vec3 hitNormal,
    const PropagateParams params
) {
    alignRayToHit(ray, hitNormal);
    return propagateRayToHit(ray.state, hitPos, hitNormal, params);
}

#endif
