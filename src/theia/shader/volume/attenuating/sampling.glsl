#ifndef _INCLUDE_VOLUME_ATTENUATING_SAMPLING
#define _INCLUDE_VOLUME_ATTENUATING_SAMPLING

//Implements the sampling strategies introduced in: C. Kulla, M. Fajardo:
//"Importance Sampling Techniques for Path Tracing in Participating Media"
//Eurographics Symposium on Rendering 2012, Vol. 31(2012), Number 4

#include "math/expm1.glsl"
#include "math/log1p.glsl"

//at very small values, switch to uniform to handle underflow correctly
//at this range, uniform and exponential are practically identical in float precision
#define _THIN_MEDIUM_LIMIT 5.9604644775390625e-8 //1 ULP

//Samples a step size proportional to exp(-mu_e * t) in the range [0, maxStep]
float sampleAttenuationNeeStepSize(
    float mu_e,                 ///< attenuation coefficient
    float maxStep,              ///< Upper sample limit
    out float prob,             ///< Sample probability
    uint idx, inout uint dim    ///< RNG state
) {
    //TODO: return maxStep for mu_e = 0?
    float u = random(idx, dim);
    float tau = mu_e * maxStep;
    if (tau < _THIN_MEDIUM_LIMIT) {
        prob = 1.0 / maxStep;
        return clamp(u * maxStep, 0.0, maxStep);
    }

    float norm = -expm1(-tau);
    float v = u * norm;
    //log(1 - v) without cancellation: log1p for small v, otherwise use
    //1 - v = (1 - u) + u*exp(-tau), a sum of two non-negative terms
    float logArg = v < 0.25 ? log1p(-v) : log((1.0 - u) + u * exp(-tau));

    float t = clamp(-logArg / mu_e, 0.0, maxStep);
    prob = mu_e * exp(-mu_e * t) / norm;
    return t;
}

//Returns the probability the given step size would have been sampled by
//sampleAttenuationNeeStepSize
float evalAttenuationNeeStepSize(
    float mu_e,
    float maxStep,
    float tStep
) {
    if (!(tStep >= 0.0 && tStep <= maxStep)) return 0.0;

    float tau = mu_e * maxStep;
    if (tau < _THIN_MEDIUM_LIMIT) return 1.0 / maxStep;
    return mu_e * exp(-mu_e * tStep) / -expm1(-tau);
}

#undef _THIN_MEDIUM_LIMIT

#endif
