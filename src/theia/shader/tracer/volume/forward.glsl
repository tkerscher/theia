layout(local_size_x = 512) in;

#include "result.glsl"
#include "material.glsl"
#include "target/common.glsl"

//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "callback.glsl"
#include "photon.glsl"
#include "source.glsl"
#include "response.glsl"
#include "target.glsl"
#include "volume.glsl"

#include "tracer/propagate/forward.glsl"
#include "util/jacobian.glsl"

uniform TraceParams {
    uint batchSize;

    uint mediumIdx;
    int objectId;

    PropagationParams propagation;
} params;

#ifndef DISABLE_NEE

void createResponse(
    ForwardRay ray,
    const TargetSample hit,
    vec3 dir,
    float weight,
    bool scattered,
    uint idx, inout uint dim
) {
    //check if we hit
    if (!hit.valid) return;

    //scatter ray if needed
    if (scattered) {
        scatter(ray, dir);
    }

    //propagate ray to hit
    ResultCode result = propagateToHit(
        ray,
        hit.position,
        hit.normal,
        false, //not sampled
        params.propagation,
        idx, dim);
    if (result < 0) return;

    //create weighted response
    ray.lin_contrib *= weight;
    HitItem item = createHit(
        ray,
        hit.objPosition,
        hit.objNormal,
        params.objectId,
        hit.worldToObj
    );
    if (item.contrib > 0.0)
        response(item, idx, dim);
}

void sampleTargetMIS(ForwardRay ray, uint idx, inout uint dim) {
    //Here we'll use the following naming scheme: pXY, where:
    // X: prob, evaluated distribution
    // Y: sampled distribution
    // T: target, P: phase
    //e.g. pTP: p_target(dir ~ phase)

    //shorthand notation
    uint med = params.mediumIdx;
    vec3 obs = ray.position;
    vec3 dir = ray.direction;

    //sample phase function
    float pPP;
    vec3 dirPhase = scatter(med, dir, random2D(idx, dim), pPP);
    TargetSample phaseHit = intersectTarget(obs, dirPhase);

    //sample target
    TargetSample targetHit = sampleTarget(obs, idx, dim);
    vec3 dirTarget = normalize(targetHit.position - obs);
    float pTT = targetHit.prob * dA_dw(obs, targetHit.position, targetHit.normal);

    //calculate cross probabilities
    float pPT = scatterProb(med, dir, dirTarget);
    float pTP = phaseHit.prob * dA_dw(obs, phaseHit.position, phaseHit.normal);

    //calculate MIS weights
    float wTarget = pTT * pPT / (pTT*pTT + pPT*pPT);
    float wPhase = pPP * pPP / (pPP*pPP + pTP*pTP);

    //create hits
    createResponse(ray, phaseHit, dirPhase, wPhase, true, idx, dim);
    createResponse(ray, targetHit, dirTarget, wTarget, true, idx, dim);
}

#endif

ResultCode trace(
    inout ForwardRay ray,
    bool allowResponse,
    uint idx, inout uint dim
) {
    //sample distance to propagate in volume
    float dist = sampleStepSize(ray, params.propagation, idx, dim);
    //trace target
    TargetSample hit = intersectTarget(ray.position, ray.direction);
    bool hitValid = hit.valid && hit.dist <= dist;
    //propagate
    dist = min(dist, hit.dist);
    ResultCode result = propagateSampled(
        ray, dist,
        hitValid,
        params.propagation,
        idx, dim);
    if (result < 0)
        return result; //abort
    
    //process hit
    if (hitValid && allowResponse) {
        alignRayToHit(ray, hit.normal);
        HitItem item = createHit(
            ray,
            hit.objPosition,
            hit.objNormal,
            params.objectId,
            hit.worldToObj
        );
        response(item, idx, dim);
        
        return RESULT_CODE_RAY_DETECTED;
    }
    else if (hitValid) {
        //hit target, but not allowed to create response -> only absorb
        return RESULT_CODE_RAY_ABSORBED;
    }

    #ifndef DISABLE_NEE
    sampleTargetMIS(ray, idx, dim);
    #endif

    //no hit
    return RESULT_CODE_SUCCESS;
}

//process macro flags
#if defined(DISABLE_NEE) && !defined(DISABLE_DIRECT_LIGHTING)
#define DIRECT_LIGHTING true
#else
#define DIRECT_LIGHTING false
#endif

#ifndef DISABLE_NEE
#define ALLOW_RESPONSE false
#else
#define ALLOW_RESPONSE true
#endif

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.batchSize)
        return;
    
    //sample ray
    ForwardRay ray = sampleLight(idx, dim);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);
    //discard ray if inside target
    if (isOccludedByTarget(ray.position)) {
        onEvent(ray, ERROR_CODE_TRACE_ABORT, idx, 0);
        return;
    }

    //try to extend first ray to target if direct lighting and MIS is enabled
    #if !defined(DISABLE_DIRECT_LIGHTING) && !defined(DISABLE_NEE)
    TargetSample directHit = intersectTarget(ray.position, ray.direction);
    createResponse(ray, directHit, ray.direction, 1.0, false, idx, dim);
    #endif

    //trace loop
    bool allowResponse = DIRECT_LIGHTING;
    for (uint i = 1; i <= PATH_LENGTH; ++i) {
        ResultCode result = trace(ray, allowResponse, idx, dim);
        bool stop = result < 0 || result == RESULT_CODE_RAY_DETECTED;
        if (!stop && i >= PATH_LENGTH) {
            //hard limit on iterations
            result = RESULT_CODE_MAX_ITER;
            stop = true;
        }
        if (!stop) {
            //interact with volume if we want to go for another iteration
            result = sampleVolumeInteraction(ray, idx, dim);
        }
        onEvent(ray, result, idx, i);
        if (stop || result < 0) return;

        //first iteration was special -> fall back to default
        allowResponse = ALLOW_RESPONSE;
    }
}
