#include "result.glsl"
#include "material.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "volume.glsl"

#include "tracer/scene/volume/index.glsl"
#include "tracer/propagate/forward.glsl"

#include "tracer/scene/forward/io.glsl"

layout(location = 0) rayPayloadInEXT TraceData traceData;

//the volume model may not support scattering. In that case all MIS contributions
//are trivially zero and we can skip them altogether.
#if !defined(VOLUME_MODEL_NO_SCATTERING) && !defined(DISABLE_NEE)

#include "tracer/scene/forward/nee.glsl"

void traceNEE(
    ForwardRay ray,
    vec3 newDir,
    float dist,
    float weight,
    inout uint dim
) {
    //scatter ray into new direction
    ResultCode result = volumeScatterRay(
        ray, newDir, gl_LaunchIDEXT.x, dim
    );
    if (result < 0) return;

    //trace nee
    traceNEE(ray, dist, weight, params.tlas, dim);
}

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

void sampleTargetMIS(ForwardRay ray, inout uint dim) {
    //Here we'll use the following naming scheme: pXY, where:
    // X: prob, evaluated distribution
    // Y: sampled distribution
    // T: target, P: phase
    //e.g. pTP: p_target(dir ~ phase)

    //shorthand notation
    vec3 obs = ray.position;
    vec3 dir = ray.direction;

    //sample phase function
    float pPP;
    vec3 dirPhase = sampleVolumeScattering(ray, pPP, gl_LaunchIDEXT.x, dim);
    //sample target guide
    TargetGuideSample targetSample = sampleTargetGuide(obs, gl_LaunchIDEXT.x, dim);
    float pTT = targetSample.prob;
    //calculate cross propabilities
    TargetGuideSample phaseSample = evalTargetGuide(obs, dirPhase);
    float pTP = phaseSample.prob;
    float pPT = volumeScatterProb(ray, targetSample.dir);

    //calculate MIS weight
    float wTarget = pTT * pTT / (pTT*pTT + pPT*pPT);
    float wPhase = pPP * pPP / (pPP*pPP + pTP*pTP);

    //trace shadow rays
    traceNEE(ray, dirPhase, phaseSample.dist, wPhase, dim);
    traceNEE(ray, targetSample.dir, targetSample.dist, wTarget, dim);
}

#endif

void main() {
    //we do not need to report rng state back as we advance it in the trace loop
    uint dim = traceData.dim;
    //propagate ray
    traceData.result = propagateSampled(
        traceData.ray,
        gl_RayTmaxEXT,
        false,
        params.propagation,
        gl_LaunchIDEXT.x, dim
    );
    if (traceData.result < 0) return;

    //optionally importance sample target guide
    #if !defined(VOLUME_MODEL_NO_SCATTERING) && !defined(DISABLE_NEE)
    sampleTargetMIS(traceData.ray, dim);
    #endif

    //sample volume interaction
    traceData.result = sampleVolumeInteraction(
        traceData.ray,
        gl_LaunchIDEXT.x,
        dim
    );
}
