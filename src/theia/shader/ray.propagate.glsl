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
    bool canScatter = ray.samples[0].constants.mu_s > 0.0;
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
ResultCode updateSamples(
    inout Ray ray,
    const float dist,
    const PropagateParams params,
    bool scatter
) {
    bool anyBelowMaxTime = false;
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        float mu_e = ray.samples[i].constants.mu_e;
        ray.samples[i].log_contrib -= mu_e * dist;
        ray.samples[i].time += dist / ray.samples[i].constants.vg;
        //time boundary check
        if (ray.samples[i].time <= params.maxTime)
            anyBelowMaxTime = true;
    }
    if (scatter) {
        [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
            ray.samples[i].lin_contrib *= ray.samples[i].constants.mu_s;
        }
    }
    //return result of boundary check
    return anyBelowMaxTime ? RESULT_CODE_SUCCESS : RESULT_CODE_RAY_DECAYED;
}

//Propagates the given ray in its direction for the given distance
//Updates its samples accordingly (see updateSamples())
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
    return updateSamples(ray, dist, params, scatter);
}

//updates ray with contribution from importance sampling
//Set hit=true, if the was stopped by a hit (affects probability calculations)
void updateRayIS(inout Ray ray, float dist, const PropagateParams params, bool hit) {
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        ray.samples[i].log_contrib += params.scatterCoefficient * dist;
    }
    if (!hit) {
        //if we hit anything, the actual prob is to travel at least dist
        // -> happens to drop the coefficient
        [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
            ray.samples[i].lin_contrib /= params.scatterCoefficient;
        }
    }
}

#endif
