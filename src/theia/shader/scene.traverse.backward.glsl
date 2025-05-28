#ifndef _INCLUDE_SCENE_TRAVERSE_BACKWARD
#define _INCLUDE_SCENE_TRAVERSE_BACKWARD

#include "ray.glsl"
#include "ray.combine.glsl"
#include "ray.propagate.glsl"
#include "ray.scatter.glsl"
#include "ray.surface.glsl"
#include "result.glsl"
#include "scene.intersect.glsl"

//user provided code
#include "rng.glsl"
#include "light.glsl"
#include "response.glsl"

/**
 * Process the surface hit the given ray produced.
*/
ResultCode processHit(
    inout BackwardRay ray,      ///< Ray that generated the hit
    const SurfaceHit hit,       ///< Surface hit
    const PropagateParams prop, ///< Propagation parameters
    float u                     ///< Random number
) {
    //propagate ray to hit
    //this also sets reference frame for polarization to plane of incidence
    ResultCode result = propagateRayToHit(
        ray, hit.worldPos, hit.rayNrm, prop
    );
    if (result < 0) {
        return result;
    }

    //process flags
    bool isAbs = (hit.flags & MATERIAL_BLACK_BODY_BIT) != 0;
    bool isDet = (hit.flags & MATERIAL_DETECTOR_BIT) != 0;
    bool volBorder = (hit.flags & MATERIAL_VOLUME_BORDER_BIT) != 0;
    bool canTransmit = (hit.flags & MATERIAL_NO_TRANSMIT_BWD_BIT) == 0;
    bool canReflect = (hit.flags & MATERIAL_NO_REFLECT_BWD_BIT) == 0;

    //get surface properties
    Reflectance surface = fresnelReflect(ray.state, hit);

    //stop tracing if ray was absorbed
    if (isAbs) return RESULT_CODE_RAY_ABSORBED;

    #ifndef DISABLE_VOLUME_BORDER
    if (volBorder) {
        crossBorder(ray, hit);
        return RESULT_CODE_VOLUME_HIT;
    }
    #endif

    //handle surface reflection/transmission
    #ifndef DISABLE_TRANSMISSION
    float r = 0.5 * (surface.r_s*surface.r_s + surface.r_p*surface.r_p);
    if (canReflect && canTransmit) {
        //importance sample what to do
        if (u < r) {
            reflectRayIS(ray, hit, surface);
        }
        else {
            transmitRayIS(ray, hit, surface);
        }
    }
    else if (canReflect) { //can only reflect
        reflectRay(ray, hit, surface); // non IS version
    }
    else if (canTransmit) { //can only transmit
        transmitRay(ray, hit, surface); // non IS version
    }
    else {
        //neither reflect nor transmit -> absorb
        result = RESULT_CODE_RAY_ABSORBED;
    }
    #else //#ifndef DISABLE_TRANSMISSION
    if (canReflect) {
        reflectRay(ray, hit, surface);
    }
    else {
        //neither reflect nor transmit -> absorb
        result = RESULT_CODE_RAY_ABSORBED;
    }
    #endif

    //done
    return result;
}

/*
 * Traces the given ray and propagates it. Creates a SurfaceHit describing if
 * a surface was hit and holds additional information about the hit.
*/
ResultCode trace(
    inout BackwardRay ray,      ///< Ray to trace using its current state
    out SurfaceHit hit,         ///< Resulting hit (includes misses)
    uint idx, inout uint dim,   ///< RNG state
    PropagateParams params      ///< Propagation parameters
) {
    //health check ray
    if (isRayBad(ray))
        return ERROR_CODE_RAY_BAD;
    vec3 dir = normalize(ray.state.direction);

    //sample distance
    float u = random(idx, dim);
    float dist = sampleScatterLength(ray, params, u);

    //trace ray against scene
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF, //mask -> hit everything
        ray.state.position,
        0.0, //t_min; self-intersections handled via offsets
        dir,
        dist);
    rayQueryProceedEXT(rayQuery);
    ResultCode queryResult = processRayQuery(ray.state, rayQuery, hit);
    //skip checking result. This allows for better debugging as the ray will
    //contain the position of where the error occurred.
    // if (result <= ERROR_CODE_MAX_VALUE)
    //     return result;
    
    //fetch actual traveled distance if we hit anything
    if (hit.valid) {
        dist = distance(ray.state.position, hit.worldPos);
    }
    
    //propagate ray
    ResultCode propResult = propagateRay(ray, dist, params);
    updateRayIS(ray, dist, params, hit.valid);

    //return earlier stop code
    return queryResult <= ERROR_CODE_MAX_VALUE ? queryResult : propResult;
    // return queryResult < 0 ? queryResult : propResult;
}

#ifndef SCENE_TRAVERSE_BACKWARD_DISABLE_SHADOW_RAYS

/**
 * Samples the light to create a shadow ray. If successful creates a hit.
*/
void traceShadowRay(
    BackwardRay ray,                ///< Ray the shadow ray is based on
    uint idx, inout uint dim,       ///< RNG state
    const CameraHit camera,         ///< Sampled camera hit
    const PropagateParams params    ///< Propagation params
) {
    //sample light
    SourceRay source = sampleLight(
        ray.state.position, vec3(0.0),
        ray.state.wavelength,
        ray.state.constants,
        idx, dim);
    //check if light is visible
    if (!isVisible(source.position, ray.state.position))
        return;
        
    //create hit by combining source and camera ray
    HitItem hit;
    ResultCode result = combineRays(ray, source, camera, params, hit);
    if (result >= 0 && hit.contrib > 0.0) {
        response(hit, idx, dim);
    }
}

#endif

/**
 * Processes the interaction indicated by hit, i.e. either surface hit or
 * volume scatter. Samples surface interaction, i.e. transmission/reflection.
*/
ResultCode processInteraction(
    inout BackwardRay ray,      ///< Ray to process
    const SurfaceHit hit,       ///< Hit to process (maybe invalid, i.e. no hit)
    const CameraHit camera,     ///< Sampled camera hit
    uint idx, inout uint dim,   ///< RNG state
    const PropagateParams params///< Propagation parameters
) {
    ResultCode result;
    //handle either intersection or scattering
    if (hit.valid) {
        result = processHit(
            ray, hit,
            params,
            random(idx, dim)
        );
        dim++;
    }
    else {
        #ifndef SCENE_TRAVERSE_BACKWARD_DISABLE_SHADOW_RAYS
        //trace shadow ray
        traceShadowRay(ray, idx, dim, camera, params);
        #endif

        //sample new direction
        scatterRay(ray, random2D(idx, dim));
        result = RESULT_CODE_RAY_SCATTERED;
    }

    return result;
}

#endif
