#ifndef _INCLUDE_TRACER_MIS
#define _INCLUDE_TRACER_MIS

#include "sphere.intersect.glsl"
#include "scatter.volume.glsl"

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
    Medium medium,
    vec3 position, vec3 direction,
    const Sphere target,
    vec2 uSphere, vec2 uPhase, //random numbers
    out float wTarget, out vec3 dirTarget,
    out float wPhase, out vec3 dirPhase
) {
    //Here we'll use the following naming scheme: pXY, where:
    // X: prob, evaluated distribution
    // Y: sampled distribution
    // S: sphere, P: phase
    //e.g. pSP: p_sphere(dir ~ phase)

    //sample target
    float pSS;
    dirTarget = sampleSphere(target, position, uSphere, pSS);
    
    //sample phase function
    float pPP;
    dirPhase = scatter(medium, direction, uPhase, pPP);

    //calculate cross propabilities
    float pSP = sampleSphereProb(target, position, dirPhase);
    float pPS = scatterProb(medium, direction, dirTarget);

    //calculate MIS weights (power heuristic)
    wTarget = pSS * pPS / (pSS*pSS + pPS*pPS);
    wPhase  = pPP * pPP / (pPP*pPP + pPS*pPS);
}

#endif
