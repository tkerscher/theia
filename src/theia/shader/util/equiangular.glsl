#ifndef _INCLUDE_UTIL_EQUIANGULAR
#define _INCLUDE_UTIL_EQUIANGULAR

//Implements the sampling strategies introduced in: C. Kulla, M. Fajardo:
//"Importance Sampling Techniques for Path Tracing in Participating Media"
//Eurographics Symposium on Rendering 2012, Vol. 31(2012), Number 4

#include "math.glsl"
#include "util/soft_dist.glsl"

//cached variables to be reused for MIS weight calculation
struct EquiAngularSampleContext {
    float d_min;
    float a;
    float theta_range;
};

/*
* Samples a stepping size from rayOrigin in rayDirection direction in the range
* [0,maxStep] proportional to 1/r^2, where r is the distance between the sampled
* point and target.
*/
float sampleEquiAngularStepSize(
    vec3 target,
    vec3 rayOrigin,
    vec3 rayDirection,
    float maxStep,
    float d_eps,
    out EquiAngularSampleContext context,
    uint idx, inout uint dim
) {
    //The equation in the paper uses tan, which explodes near +/- pi/2
    //Instead we use this equivalent formula:
    //
    //               r^2 * sin(theta)
    // t(u) = ----------------------------- ; theta ~ U(theta_a, theta_b)
    //         D*cos(theta) + a*sin(theta)

    vec3 o2t = target - rayOrigin;
    float a = dot(o2t, rayDirection);
    float D = length(crosser(o2t, rayDirection));
    float r2 = dot(o2t, o2t);
    context.d_min = D; //use unsoftened version so we can branch later
    context.a = a;
    //we use a softening kernel to avoid division by zero later
    if (D < d_eps) {
        D = softDist_comp(D, d_eps);
        r2 = D*D + a*a; //avoid this source of rounding if not needed
    }
    float theta_range = atan(D * maxStep, r2 - a * maxStep);
    context.theta_range = theta_range;

    float theta = random(idx, dim) * theta_range;
    float denom = D * cos(theta) + a * sin(theta);
    //denom can be zero for theta=pi/2 giving 0/0, which corresponds to the far end
    float t = denom > 0.0 ? r2 * sin(theta) / denom : maxStep;
    return clamp(t, 0.0, maxStep); //guarding against numeric errors
}

float equiAngularMisWeight(
    vec3 target,
    vec3 rayPosition,
    float d_eps,
    float t_step,
    float residualProb,
    const EquiAngularSampleContext context
) {
    vec3 o2t = target - rayPosition;
    float r2 = dot(o2t, o2t);
    //calculate contribution weight
    //    Phi              1            Phi             T_b - T_a              Phi
    // --------- * ----------------- = ----- * ---------------------------- = ----- * weight
    //  4pi*r^2     p_L(t) + p_T(t)     4pi     D + (T_b - T_a)*r^2*p_T(t)     4pi

    //did we soften?
    if (context.d_min >= d_eps) {
        //no softening
        return context.theta_range / (context.d_min + context.theta_range * r2 * residualProb);
    }
    else {
        //a softened D changes p_L causing some terms to no longer cancel
        float D = softDist_comp(context.d_min, d_eps);
        float s = t_step - context.a;
        float q = D * D + s * s;
        float eps2 = d_eps * d_eps;
        return softInvSquare_comp(r2, eps2) * context.theta_range * q /
            (D + context.theta_range * q * residualProb);
    }
}

#endif
