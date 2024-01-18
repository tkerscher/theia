#ifndef _INCLUDE_RAY_PROPAGATE
#define _INCLUDE_RAY_PROPAGATE

#include "ray.glsl"
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
//Set hit=true, if the was stopped by a hit (affects probability calculations)
//Returns true, if the ray can proceed, i.e. is within boundaries.
bool updateSamples(
    inout Ray ray,
    const float dist,
    const PropagateParams params,
    bool scatter, bool hit
) {
    bool anyBelowMaxTime = false;
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        float mu_e = ray.samples[i].constants.mu_e;
        ray.samples[i].log_contrib += params.scatterCoefficient * dist - mu_e * dist;
        ray.samples[i].time += dist / ray.samples[i].constants.vg;
        if (scatter)
            ray.samples[i].lin_contrib *= ray.samples[i].constants.mu_s;
        if (!hit)
            ray.samples[i].lin_contrib /= params.scatterCoefficient;
        //time boundary check
        if (ray.samples[i].time <= params.maxTime)
            anyBelowMaxTime = true;
    }
    //return result of boundary check
    return anyBelowMaxTime;
}

//Propagates the given ray in its direction for the given distance
//Updates its samples accordingly (see updateSamples())
//Set scatter=true, if mu_s should be applied to lin_contrib
//Set hit=true, if the was stopped by a hit (affects probability calculations)
//Returns true, if the ray can proceed, i.e. is within boundaries.
bool propagateRay(
    inout Ray ray,
    float dist,
    const PropagateParams params,
    bool scatter, bool hit
) {
    //update ray
    ray.position += dist * normalize(ray.direction);
    //boundary check
    bool outside =
        any(lessThan(ray.position, params.lowerBBoxCorner)) ||
        any(greaterThan(ray.position, params.upperBBoxCorner));
    if (CHECK_BRANCH(outside)) {
        return false;
    }

    //update samples
    return updateSamples(ray, dist, params, scatter, hit);
}

#endif
