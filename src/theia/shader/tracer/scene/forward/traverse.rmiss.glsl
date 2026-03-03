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

//MIS: sample both phase function & detector
//
//  w_X(X)            p_X(X)
// -------- = ---------------------
//  p_X(X)     p_X(X)^2 + p_Y(X)^2
//
//  ^^^^^^ MIS weight divided by IS probability

void sampleTargetMIS(ForwardRay ray, inout uint dim) {
    //Here we'll use the following naming scheme: pXY, where:
    // X: prob, evaluated distribution
    // Y: sampled distribution
    // T: target, P: phase
    //e.g. pTP: p_target(dir ~ phase)

    //sample volume scattering
    float pPP;
    vec3 dirPhase = sampleVolumeScattering(ray, pPP, gl_LaunchIDEXT.x, dim);
    TargetGuideSample phaseSample = evalTargetGuide(ray.position, dirPhase);
    //sample target guide
    TargetGuideSample targetSample = sampleTargetGuide(ray.position, gl_LaunchIDEXT.x, dim);
    vec3 dirTarget = targetSample.dir;
    float pTT = targetSample.prob;
    //calculate cross probabilities
    float pPT = volumeScatterProb(ray, dirTarget);
    float pTP = phaseSample.prob;

    //calculate MIS weights
    float wPhase = pPP / (pPP*pPP + pTP*pTP);
    float wTarget = pTT / (pTT*pTT + pPT*pPT);

    //trace shadow rays
    traceNEE(ray, dirPhase, phaseSample.dist, wPhase, dim);
    traceNEE(ray, dirTarget, targetSample.dist, wTarget, dim);
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
