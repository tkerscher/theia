#ifndef _INCLUDE_RAY_PROPAGATE
#define _INCLUDE_RAY_PROPAGATE

#include "ray.glsl"
#include "result.glsl"
#include "util.branch.glsl"

struct PropagateParams {
    float scatterCoefficient;

    vec3 lowerBBoxCorner;
    vec3 upperBBoxCorner;
    float maxTime;

    float maxDist;
};

//samples the distance until the next scatter event
//handles non scattering media correctly, where it returns params.maxDist instead
float sampleScatterLength(const Ray ray, const PropagateParams params, float u) {
    float dist = params.maxDist;
    bool canScatter = ray.constants.mu_s > 0.0;
    if (CHECK_BRANCH(canScatter)) {
        //sample exponential distribution
        //use u -> 1.0 - u > 0.0 to be safe on the log
        dist = -log(1.0 - u) / params.scatterCoefficient;
    }
    return dist;
}

//Updates samples in ray as if they traveled a given distance.
//Includes sampling propabilities and check for maxTime
//Set scatter=true, if mu_s should be applied to lin_contrib
ResultCode updateRay(
    inout Ray ray,
    const float dist,
    const PropagateParams params,
    bool scatter
) {
    ray.log_contrib -= ray.constants.mu_e * dist;
    ray.time += dist / ray.constants.vg;
    if (scatter) {
        ray.lin_contrib *= ray.constants.mu_s;
    }

    //return result of boundary check
    return ray.time <= params.maxTime ? RESULT_CODE_SUCCESS : RESULT_CODE_RAY_DECAYED;
}

//Propagates the given ray in its direction for the given distance
//Updates its samples accordingly (see updateRay())
//Set scatter=true, if mu_s should be applied to lin_contrib
ResultCode propagateRay(
    inout Ray ray,
    float dist,
    const PropagateParams params,
    bool scatter
) {
    //update ray
    ray.position += dist * ray.direction;
    //boundary check
    bool outside =
        any(lessThan(ray.position, params.lowerBBoxCorner)) ||
        any(greaterThan(ray.position, params.upperBBoxCorner));
    if (CHECK_BRANCH(outside)) {
        return RESULT_CODE_RAY_LOST;
    }

    //update samples
    return updateRay(ray, dist, params, scatter);
}

//updates ray with contribution from importance sampling
//Set hit=true, if the was stopped by a hit (affects probability calculations)
void updateRayIS(inout Ray ray, float dist, const PropagateParams params, bool hit) {
    //We only used IS, if scattering is possible
    //(otherwise we used a constant maxDistance)
    bool canScatter = ray.constants.mu_s > 0.0;
    if (!canScatter)
        return;

    ray.log_contrib += params.scatterCoefficient * dist;
    if (!hit) {
        //if we hit anything, the actual prob is to travel at least dist
        // -> happens to drop the coefficient
        ray.lin_contrib /= params.scatterCoefficient;
    }
}

#endif
