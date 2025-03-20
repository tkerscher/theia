#ifndef _INCLUDE_SCENE_TRAVERSE
#define _INCLUDE_SCENE_TRAVERSE

#include "ray.glsl"
#include "ray.propagate.glsl"
#include "ray.response.glsl"
#include "ray.scatter.glsl"
#include "ray.surface.glsl"
#include "result.glsl"
#include "scene.intersect.glsl"

//user provided code
#include "rng.glsl"
#include "response.glsl"
#ifndef SCENE_TRAVERSE_DISABLE_MIS
#include "target_guide.common.glsl"
#include "target_guide.glsl"
#endif

#if defined(SCENE_TRAVERSE_FORWARD) && !defined(SCENE_TRAVERSE_BACKWARD)
#define RAY ForwardRay
#define MATERIAL_TARGET_BIT MATERIAL_DETECTOR_BIT
#elif !defined(SCENE_TRAVERSE_FORWARD) && defined(SCENE_TRAVERSE_BACKWARD)
#define RAY BackwardRay
#define MATERIAL_TARGET_BIT MATERIAL_LIGHT_SOURCE_BIT
#else
#error "No scene traverse direction specified!"
#endif

/**
 * Create a response from the given ray and surface hit
*/
void createResponse(
    RAY ray,                            ///< Ray that generated that hit
    const SurfaceHit hit,               ///< Surface hit
    const Reflectance surface,          ///< Surface properties
    bool absorb                         ///< True, if the surface absorbs the ray
) {
    //if target is not absorbing, we have to emulate a transmission before the
    //response as we have to subtract the reflected part.
    if (!absorb) {
        #ifdef HIT_USE_TRANSMITTED_DIR

        //use the transmitted ray for hit
        transmitRay(ray, hit, surface);

        #else

        //call transmitRay for transmission calculation, but restore incident dir
        vec3 dir = ray.state.direction;
        transmitRay(ray, hit, surface);
        ray.state.direction = dir;

        #endif
    }

    //create response
    HitItem item = createHit(ray, hit.objPos, hit.objNrm, hit.worldToObj);
    if (item.contrib > 0.0)
        response(item);
}

/**
 * Process the surface hit the given ray produced.
*/
ResultCode processHit(
    inout RAY ray,                  ///< Ray that generated the hit
    const SurfaceHit hit,           ///< Surface hit
    const PropagateParams params,   ///< Propagation parameters
    uint targetId,                  ///< Id of target
    float u,                        ///< Random number
    bool allowResponse              ///< Whether a detector hit should create a response
) {
    //propagate ray to hit
    //this also sets reference frame for polarization to plane of incidence
    ResultCode result = propagateRayToHit(
        ray, hit.worldPos, hit.rayNrm, params
    );
    if (result < 0) {
        return result;
    }
    //Set default code
    result = RESULT_CODE_RAY_HIT;

    //process flags
    bool isAbs = (hit.flags & MATERIAL_BLACK_BODY_BIT) != 0;
    bool isTarget = (hit.flags & MATERIAL_TARGET_BIT) != 0;
    bool volBorder = (hit.flags & MATERIAL_VOLUME_BORDER_BIT) != 0;
    bool canTransmit = (hit.flags & MATERIAL_NO_TRANSMIT_BWD_BIT) == 0;
    bool canReflect = (hit.flags & MATERIAL_NO_REFLECT_BWD_BIT) == 0;

    //get surface properties
    Reflectance surface = fresnelReflect(ray.state, hit);

    //create response if allowed
    if (allowResponse && isTarget && hit.customId == targetId) {
        createResponse(ray, hit, surface, isAbs);
        //do not return early as the ray is allowed to carry on
        //(e.g. partially reflect)
        result = RESULT_CODE_RAY_DETECTED;
    }

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

/**
 * Processes the result of a given shadow ray hitting a surface and produces a
 * response if applicable.
*/
void processShadowRay(
    RAY ray,                        ///< Shadow ray
    const SurfaceHit hit,           ///< Surface hit of shadow ray
    uint targetId,                  ///< Id of target
    const PropagateParams params    ///< Propagation parameters
) {
    //check if we hit target
    bool isTarget = (hit.flags & MATERIAL_TARGET_BIT) != 0;
    if (!hit.valid || hit.customId != targetId || !isTarget) return; //hit something else
    //propagate ray to hit
    if (propagateRayToHit(ray, hit.worldPos, hit.rayNrm, params) < 0) return;

    //get surface properties
    Reflectance surface = fresnelReflect(ray.state, hit);

    //create response
    bool black = (hit.flags & MATERIAL_BLACK_BODY_BIT) != 0;
    createResponse(ray, hit, surface, black);
}

/**
 * Traces an independent shadow ray based on the given one in a new direction.
*/
void traceShadowRay(
    RAY ray,                        ///< Ray the shadow one is based on
    vec3 dir,                       ///< Direction of shadow ray
    float dist,                     ///< Max distance of shadow ray
    uint targetId,                  ///< Id of target
    const PropagateParams params,   ///< Propagation parameters
    float weight                    ///< Weight of shadow ray applied to its hits
) {
    //scatter local ray into dir (takes care of polarization)
    scatterRayIS(ray, dir);

    //trace ray against scene
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF, //mask -> hit anything
        ray.state.position,
        0.0, dir, dist);
    rayQueryProceedEXT(rayQuery);

    //process hit
    SurfaceHit hit;
    if (processRayQuery(ray.state, rayQuery, hit) < 0) return; //hit?
    
    //process shadow ray
    ray.state.lin_contrib *= weight; //apply weight here
    processShadowRay(ray, hit, targetId, params);
}

#ifndef SCENE_TRAVERSE_DISABLE_MIS

//MIS is a sampling method that combines multiple distributions using weights
//to minimize variance increase. Allows to use specialized distributions (here
//sampling the target sphere) to increase performance. Distributions need to
//cover the variable space only jointly, i.e. they are allowed to assign zero
//probability to a valid value as long as there is at least one that can sample
//it

//MIS: sample both scattering phase function & detector
//also include factors of phase function and sample propability:
//             p_XX^2        p_PX   <- scattering phase function
// w_X = ---------------- * ------
//        p_XX^2 + p_YX^2    p_XX   <- importance sampling
//       \-------V------/
//          MIS weight
//to improve precision, we already reduce the fraction where possible

void sampleTargetMIS(
    RAY ray,
    uint targetId,
    uint idx, inout uint dim,
    const PropagateParams params
) {
    //Here we'll use the following naming scheme: pXY, where:
    // X: prob, evaluated distribution
    // Y: sampled distribution
    // T: target, P: phase
    //e.g. pTP: p_target(dir ~ phase)

    //shorthand notation
    Medium med = Medium(ray.state.medium);
    vec3 obs = ray.state.position;
    vec3 dir = ray.state.direction;

    //sample phase function
    float pPP;
    vec3 dirPhase = scatter(med, dir, random2D(idx, dim), pPP);
    //sample target guide
    TargetGuideSample targetSample = sampleTargetGuide(obs, idx, dim);
    float pTT = targetSample.prob;
    //calculate cross probabilities
    TargetGuideSample phaseSample = evalTargetGuide(obs, dirPhase);
    float pTP = phaseSample.prob;
    float pPT = scatterProb(med, dir, targetSample.dir);

    //calculate MIS weights
    float wTarget = pTT * pPT / (pTT*pTT + pPT*pPT);
    float wPhase = pPP * pPP / (pPP*pPP + pTP*pTP);

    //trace shadow rays
    traceShadowRay(ray, dirPhase, phaseSample.dist, targetId, params, wPhase);
    traceShadowRay(ray, targetSample.dir, targetSample.dist, targetId, params, wTarget);
}

#endif

/*
 * Traces the given ray and propagates it. Creates a SurfaceHit describing if
 * a surface was hit and holds additional information about the hit.
 *
 * If MIS is enabled and allowResponse=true, tries to extend the ray to the
 * detector to create a response.
*/
ResultCode trace(
    inout RAY ray,                  ///< Ray to trace using its current state
    out SurfaceHit hit,             ///< Resulting hit (includes misses)
    uint targetId,                  ///< Id of target
    uint idx, inout uint dim,       ///< RNG state
    const PropagateParams params,   ///< Propagation parameters
    bool allowResponse              ///< Whether detector hits can create a response
) {
    //health check ray
    if (isRayBad(ray))
        return ERROR_CODE_RAY_BAD;
    vec3 dir = normalize(ray.state.direction);

    //sample distance
    float dist = sampleScatterLength(ray, params, random(idx, dim));

    //next event estimate target by extending ray if possible
    //only if allowResponse is true
    #ifndef SCENE_TRAVERSE_DISABLE_MIS
    float sampledDist = dist;
    TargetGuideSample guideSample = evalTargetGuide(ray.state.position, dir);
    //check if we have a chance on hitting the target by extending the ray
    bool mis_target = allowResponse && guideSample.prob > 0.0;
    if (mis_target) {
        //check if we actually extended the ray
        //disable shadow ray if not
        mis_target = guideSample.dist > dist;
        dist = max(guideSample.dist, dist);
    }
    #endif

    //trace ray against scene
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF, //mask -> hit everything
        ray.state.position,
        0.0, //t_min; self-intersections handled via offsets
        dir,
        dist
    );
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

    #ifndef SCENE_TRAVERSE_DISABLE_MIS
    //Check if hit was actually our shadow ray
    if (mis_target && hit.valid && dist > sampledDist && queryResult >= 0) {
        //create response
        processShadowRay(ray, hit, targetId, params);
        //act like we didn't actually hit anything
        hit.valid = false;
        dist = sampledDist;
    }
    #endif

    //propagate ray
    ResultCode propResult = propagateRay(ray, dist, params);
    updateRayIS(ray, dist, params, hit.valid);

    //return earlier stop code
    return queryResult <= ERROR_CODE_MAX_VALUE ? queryResult : propResult;
    // return queryResult < 0 ? queryResult : propResult;
}

/**
 * Processes the interaction indicated by hit, i.e. either surface hit or
 * volume scatter.
 *  - Samples surface interaction, i.e. transmission/reflection.
 *  - Detector response enabled via allowResponse.
 *  - If MIS enabled and allowResponse, samples detector to create shadow rays
*/
ResultCode processInteraction(
    inout RAY ray,                  ///< Ray to process
    const SurfaceHit hit,           ///< Hit to process (maybe invalid, i.e. no hit)
    uint targetId,                  ///< Id of target
    uint idx, inout uint dim,       ///< RNG state
    const PropagateParams params,   ///< Propagation parameters
    bool allowResponse,             ///< Whether detector hit by chance are allowed to create a response
    bool last                       ///< True on last iteration (skips MIS)
) {
    ResultCode result;
    //handle either intersection or scattering
    if (hit.valid) {
        result = processHit(
            ray, hit,
            params,
            targetId,
            random(idx, dim),
            allowResponse
        );
    }
    else {
        //do not bother scattering on the last iteration (we wont hit anything)
        //this also prevents MIS to sample paths one longer than PATH_LENGTH
        if (!last) {
            #ifndef SCENE_TRAVERSE_DISABLE_MIS
            //trace shadow ray
            sampleTargetMIS(ray, targetId, idx, dim, params);
            #endif

            //scatter ray in new direction
            scatterRay(ray, random2D(idx, dim));
        }
        result = RESULT_CODE_RAY_SCATTERED;
    }

    return result;
}

#endif
