/*
Forward and backward propagation uses the same code, but since ForwardRay and
BackwardRay may behave differently, we need separate functions. Unfortunately,
GLSL does not support templates, so we use macros instead.
*/

#ifndef RAY
#error "RAY template argument not defined"
#endif

void alignRayToHit(
    inout RAY ray,
    vec3 position,
    vec3 normal
) {
    ray.position = position;
    alignRayToHit(ray, normal);
}

ResultCode checkBoundary(const RAY ray, const PropagationParams params) {
    bool outside =
        any(lessThan(ray.position, params.lowerBBoxCorner)) ||
        any(greaterThan(ray.position, params.upperBBoxCorner));
    if (outside)
        return RESULT_CODE_RAY_LOST;
    
    #ifdef RAY_TRANSIENT
    if (ray.time > params.maxTime)
        return RESULT_CODE_RAY_DECAYED;
    #endif

    return RESULT_CODE_SUCCESS;
}

//If the ray constitutes a particle, it does not posses a contrib parameter.
//In that case we must not produce code using that parameter.

#ifndef RAY_PARTICLE

void scatter(inout RAY ray, vec3 dir) {
    volumeScatterRay(ray, dir);
    scatterRay(ray, dir);
}

float sampleStepSize(
    const RAY ray,
    const PropagationParams params,
    uint idx, inout uint dim
) {
    float dist = params.maxDist;
    //negative or NaN sample coefficients denotes importance sampling the volume model
    //mind that comparison with NaN always return false
    if (!(params.sampleCoefficient >= 0.0)) {
        dist = min(dist, sampleInteractionLength(ray, idx, dim));
    }
    else if (params.sampleCoefficient > 0.0) {
        //sample exponential distribution
        //use u -> 1.0 - u > 0.0 to be safe on the log
        float u = random(idx, dim);
        dist = min(dist, -log(1.0 - u) / params.sampleCoefficient);
    }
    //remaining case; sampleCoefficient = 0.0 -> disable volume interactions
    return dist;
}

ResultCode updateRayIS(
    inout RAY ray,
    float dist,
    bool hit,
    const PropagationParams params,
    uint idx, inout uint dim
) {
    //negative or NaN sample coefficients denotes importance sampling the volume model
    //mind that comparison with NaN always return false
    if (!(params.sampleCoefficient >= 0.0)) {
        return applyVolumeSampled(ray, dist, hit, idx, dim);
    }
    else if (params.sampleCoefficient > 0.0) {
        //we sampled exponential distribution
        ray.log_contrib += params.sampleCoefficient * dist;
        if (!hit) {
            //if we hit anything, the actual prop is to travel at least dist
            // -> happens to cancel the coefficient
            // -> we need to divide by the scatter coef if we did not hit anything
            ray.lin_contrib /= params.sampleCoefficient;
        }
    }
    //If sampleCoefficient is zero, we didn't do any importance sampling
    // -> do nothing
    return RESULT_CODE_SUCCESS;
}

ResultCode propagate(
    inout RAY ray,
    float dist,
    bool hit,
    bool sampled,
    const PropagationParams params,
    uint idx, inout uint dim
) {
    //if we have limited the propagation distance, it's the same as hitting something
    if (dist >= params.maxDist)
        hit = true;

    //propagate ray itself
    ResultCode code = propagateRay(ray, dist);
    if (code < 0) return code;
    //apply volume effects
    if (sampled) {
        code = updateRayIS(ray, dist, hit, params, idx, dim);
    }
    else {
        code = applyVolume(ray, dist, hit, idx, dim);
    }
    if (code < 0) return code;

    //boundary check
    return checkBoundary(ray, params);
}

ResultCode propagateToHit(
    inout RAY ray,
    vec3 position,
    vec3 normal,
    bool sampled,
    const PropagationParams params,
    uint idx, inout uint dim
) {
    //first normal propagation
    float dist = distance(ray.position, position);
    ResultCode result = propagate(ray, dist, true, sampled, params, idx, dim);
    //then align
    alignRayToHit(ray, position, normal);

    return result;
}

#else //#ifndef RAY_PARTICLE

float sampleStepSize(
    const RAY ray,
    const PropagationParams params,
    uint idx, inout uint dim
) {
    return min(
        params.maxDist,
        sampleInteractionLength(ray, idx, dim)
    );
}

ResultCode updateRayIS(
    inout RAY ray,
    float dist,
    bool hit,
    const PropagationParams params,
    uint idx, inout uint dim   
) {
    return applyVolumeSampled(ray, dist, hit, idx, dim);
}

#endif

//propagates given ray if the propagation distance has been sampled
ResultCode propagateSampled(
    inout RAY ray,
    float dist,
    bool hit,
    const PropagationParams params,
    uint idx, inout uint dim
) {
    //if we have limited the propagation distance, it's the same as hitting something
    if (dist >= params.maxDist)
        hit = true;

    //propagate ray itself
    ResultCode code = propagateRay(ray, dist);
    if (code < 0) return code;
    //apply volume effects
    code = updateRayIS(ray, dist, hit, params, idx, dim);
    if (code < 0) return code;

    //boundary check
    return checkBoundary(ray, params);
}

ResultCode propagateSampledToHit(
    inout RAY ray,
    vec3 position,
    vec3 normal,
    const PropagationParams params,
    uint idx, inout uint dim
) {
    //first normal propagation
    float dist = distance(ray.position, position);
    ResultCode result = propagateSampled(ray, dist, true, params, idx, dim);
    //then align
    alignRayToHit(ray, position, normal);

    return result;
}
