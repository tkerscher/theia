#include "result.glsl"
#include "material.glsl"
#include "scene/types.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "surface.glsl"
#include "response.glsl"

#include "tracer/scene/target/config.glsl"
#include "tracer/scene/volume/proxy.glsl"
#include "tracer/scene/target/propagate.glsl"
#include "scene/intersect.glsl"

#include "tracer/scene/target/io.glsl"

#ifndef DISABLE_NEE
#include "tracer/scene/target/nee.glsl"
#endif

//mapping from TLAS instance -> objectId
readonly buffer ObjectIdMap{ int objectIdMap[]; };

layout(location = 0) rayPayloadInEXT TraceData traceData;
hitAttributeEXT vec2 barys; //filled by default intersection shader

//the surface model may not support (non specular) scattering. In that case all
//MIS contributions are trivially zero and we can skip them alltogether
#if !defined(SURFACE_MODEL_SPECULAR) && !defined(DISABLE_NEE)

void traceNEE(
    TRACE_RAY ray,
    const SurfaceHit hit,
    #ifdef SurfaceProperties
    const SurfaceProperties props,
    #endif
    vec3 newDir,
    float dist,
    float weight,
    inout uint dim
) {
    //scatter ray in new direction
    ResultCode result = surfaceScatterRay(
        ray, hit,
        #ifdef SurfaceProperties
        props,
        #endif
        newDir,
        gl_LaunchIDEXT.x, dim
    );
    if (result < 0) return;

    //extra cosine from geometric term (projected area in radiance definition)
    ray.lin_contrib *= abs(dot(newDir, hit.rayNrm));

    //trace NEE
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

void sampleTargetMIS(
    TRACE_RAY ray,
    const SurfaceHit hit,
    #ifdef SurfaceProperties
    const SurfaceProperties props,
    #endif
    inout uint dim
) {
    //Here we'll use the following naming scheme: pXY, where:
    // X: prob, evaluated distribution
    // Y: sampled distribution
    // T: target, P: phase
    //e.g. pTP: p_target(dir ~ phase)

    //sample surface scattering
    float pPP;
    vec3 dirPhase = sampleSurfaceScattering(
        ray, hit,
        #ifdef SurfaceProperties
        props,
        #endif
        pPP,
        gl_LaunchIDEXT.x, dim
    );
    TargetGuideSample phaseSample = evalTargetGuide(ray.position, dirPhase);
    //sample target guide
    TargetGuideSample targetSample = sampleTargetGuide(ray.position, gl_LaunchIDEXT.x, dim);
    vec3 dirTarget = targetSample.dir;
    float pTT = targetSample.prob;
    //calculate cross probabilities
    float pPT = surfaceScatterProb(
        ray, hit,
        #ifdef SurfaceProperties
        props,
        #endif
        dirTarget
    );
    float pTP = phaseSample.prob;

    //calculate MIS weights
    float wPhase = pPP / (pPP*pPP + pTP*pTP);
    float wTarget = pTT / (pTT*pTT + pPT*pPT);

    //trace shadow rays
    if (pPP > 0.0) {
        traceNEE(
            ray, hit,
            #ifdef SurfaceProperties
            props,
            #endif
            dirPhase, phaseSample.dist,
            wPhase, dim
        );
    }
    if (pTT > 0.0) {
        //Only trace this NEE if it has positive weight
        //since both pTT and pPT might be zero, wTarget can become NaN
        //thus this extra check (traceNEE also checks for zero weight)
        traceNEE(
            ray, hit,
            #ifdef SurfaceProperties
            props,
            #endif
            dirTarget, targetSample.dist,
            wTarget, dim
        );
    }
}

#endif

void main() {
    //we do not need to report rng state back as we advance it in the trace loop
    uint dim = traceData.dim;
    //resolve
    SurfaceHit hit;
    traceData.result = resolveIntersection(traceData.ray.mediumIdx, barys, hit);
    if (traceData.result < 0) return;

    //propagate ray to hit
    traceData.result = propagateSampledToHit(
        traceData.ray,
        hit.worldPos,
        hit.rayNrm,
        params.propagation,
        gl_LaunchIDEXT.x, dim
    );
    if (traceData.result < 0) return;

    //if the surface model requests it, prepare interaction
    #ifdef SurfaceProperties
    SurfaceProperties props = prepareSurface(
        traceData.ray, hit, gl_LaunchIDEXT.x, dim
    );
    #endif

    //create response if we hit target and the tracer allows it
    //if we do NEE, we will create responses exclusively through it and not in
    //the tracing loop. Otherwise, do it here.
    #ifdef DISABLE_NEE
    //did we hit a target?
    bool isTarget = (hit.flags & MATERIAL_TARGET_BIT) != 0;
    //do we have a filter on the objectId? (0x80000000 marks no filter)
    int objectId = objectIdMap[gl_InstanceID];
    bool filtered = params.targetId != 0x80000000 && params.targetId != objectId;
    if (isTarget && !filtered) {
        HitItem item;
        bool success = processSurfaceTargetHit(
            traceData.ray,
            hit,
            #ifdef SurfaceProperties
            props,
            #endif
            objectId,
            item,
            gl_LaunchIDEXT.x, dim
        );
        if (success) {
            response(item, gl_LaunchIDEXT.x, dim);
        }
    }
    #endif

    //skip surface sampling if it was marked as absorber and return early
    //note that absorber may still create a response, so do not skip that step
    if ((hit.flags & MATERIAL_BLACK_BODY_BIT) != 0) {
        traceData.result = RESULT_CODE_RAY_ABSORBED;
        return;
    }

    //optionally, importance sample target guide for MIS NEE
    #if !defined(SURFACE_MODEL_SPECULAR) && !defined(DISABLE_NEE)
    sampleTargetMIS(
        traceData.ray, hit,
        #ifdef SurfaceProperties
        props,
        #endif
        dim  
    );
    #endif

    //sample surface interaction
    traceData.result = sampleSurfaceInteraction(
        traceData.ray,
        hit,
        #ifdef SurfaceProperties
        props,
        #endif
        gl_LaunchIDEXT.x, dim
    );
    if (traceData.result < 0) return;

    //if we have a specular surface, we can still do NEE by extending the
    //already processed ray to the target guide
    #if defined(SURFACE_MODEL_SPECULAR) && !defined(DISABLE_NEE)
    traceNEE(traceData.ray, params.tlas, dim);
    #endif
}
