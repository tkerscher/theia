#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require

//check expected macros
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef N_LAMBDA
#error "N_LAMBDA not defined"
#endif
#ifndef N_SCATTER
#error "N_SCATTER not defined"
#endif
#ifndef DIM_OFFSET
#error "DIM_OFFSET not defined"
#endif
// #samples per iteration
#define DIM_STRIDE 5

layout(local_size_x = BLOCK_SIZE) in;

#include "material.glsl"
#include "scatter.volume.glsl"
#include "sphere.intersect.glsl"

#include "lightsource.common.glsl"
#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "source.glsl"
#include "response.glsl"

struct Sample {
    float wavelength;
    float time;
    float lin_contrib;
    float log_contrib;
    MediumConstants constants;
};
struct Ray {
    vec3 position;
    vec3 direction;
    Sample samples[N_LAMBDA];
};

layout(scalar) uniform TraceParams {
    Sphere target;
    float scatterCoefficient;

    uvec2 medium;

    vec3 lowerBBoxCorner;
    vec3 upperBBoxCorner;
    float maxTime;
} params;

//updates the ray after traversing
//returns true if the ray still is inside the trace boundaries
bool updateRay(inout Ray ray, vec3 dir, float dist, bool hit) {
    //update ray
    ray.position += dir * dist;
    //boundary check
    bool outside =
        any(lessThan(ray.position, params.lowerBBoxCorner)) ||
        any(greaterThan(ray.position, params.upperBBoxCorner));
    if (subgroupAll(outside)) {
        return false;
    }
    else if (outside) {
        //mixed branching
        return false;
    }

    //update all samples
    float lambda = params.scatterCoefficient;
    bool anyBelowMaxTime = false;
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        float mu_e = ray.samples[i].constants.mu_e;
        ray.samples[i].log_contrib += lambda * dist - mu_e * dist;
        ray.samples[i].time += dist / ray.samples[i].constants.vg;
        if (!hit) ray.samples[i].lin_contrib != lambda;
        //time boundary check
        if (ray.samples[i].time <= params.maxTime)
            anyBelowMaxTime = true;
    }
    //return result of boundary check
    return anyBelowMaxTime;
}

//traces ray
//returns true, if target was hit and tracing should be stopped
bool trace(inout Ray ray, uint idx, uint dim) {
    //sample distance
    float u = random(idx, dim); dim++;
    float dist = -log(1.0 - u) / params.scatterCoefficient;

    //trace sphere
    vec3 dir = normalize(ray.direction); //just to be safe
    float t = intersectSphere(params.target, ray.position, dir);

    //check if we hit
    bool hit = t <= dist;
    //update ray
    bool good = updateRay(ray, dir, hit ? t : dist, hit);
    //check if trace is still in bounds
    if (subgroupAll(!good)) {
        return true;
    }
    else if (!good) {
        //mixed branching
        return true;
    }

    //do either:
    // - create a hit item if we hit target
    // - volume scatter + MIS target
    //either way, we'll create hits on the queue

    vec3 pos, nrm;
    HitItem hits[N_LAMBDA];
    
    if (hit) {
        //create normal
        nrm = normalize(ray.position - params.target.position);
        //transform pos to sphere local coords / recalc to improve float error
        pos = nrm * params.target.radius;
        [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
            hits[i] = HitItem(
                pos, dir, nrm,
                ray.samples[i].wavelength,
                ray.samples[i].time,
                exp(ray.samples[i].log_contrib) * ray.samples[i].lin_contrib
            );
        }
    }
    else {
        /***************************************************************************
        * MIS: sample both scattering phase function & detector                   *
        *                                                                         *
        * We'll use the following naming scheme: pXY, where                       *
        * X: prob, distribution                                                   *
        * Y: sampled distribution                                                 *
        * S: scatter, D: detector                                                 *
        * e.g. pDS: p_det(dir ~ scatter)                                          *
        **************************************************************************/

        //sample detector
        vec2 rng = random2D(idx, dim); dim += 2;
        float pDD, detDist;
        vec3 detDir = sampleSphere(params.target, ray.position, rng, detDist, pDD);
        detDist = intersectSphere(params.target, ray.position, detDir);
        //create hit params
        //in the rare case, that we dit not hit the target: detDist = +inf
        //it's easier programming wise to check for this later and ignore the
        //corresponding hit. Due to branching there should not be any performance hit
        pos = ray.position + detDir * detDist;
        dir = detDir;
        nrm = normalize(pos - params.target.position);
        //transform pos to sphere local coords / recalc to improve float error
        pos = nrm * params.target.radius;

        //sample scatter phase function
        Medium medium = Medium(params.medium);
        rng = random2D(idx, dim); dim += 2;
        float pSS;
        vec3 scatterDir = scatter(medium, ray.direction, rng, pSS);

        //calculate cross probs pSD, pDS
        float pSD = scatterProb(medium, ray.direction, detDir);
        float pDS = sampleSphereProb(params.target, ray.position, scatterDir);
        //calculate MIS weights
        float w_scatter = pSS*pSS / (pSS*pSS + pSD*pSD);
        float w_det = pDD*pSD / (pDD*pDD + pDS*pDS);
        //^^^ For the detector weight, two effects happen: attenuation due to phase
        //    phase function (= pSD) and canceling of sampled distribution:
        //      f(x)*phase(theta)/pDD * w_det = f(x)*pSD/pDD * w_det
        //    Note that mu_s was already applied

        //update ray
        ray.direction = scatterDir;
        //iterate samples to update and create hits
        [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
            ray.samples[i].lin_contrib *= ray.samples[i].constants.mu_s;
            //create hits
            float mu_e = ray.samples[i].constants.mu_e;
            float contrib = exp(ray.samples[i].log_contrib - mu_e * dist);
            contrib *= ray.samples[i].lin_contrib * w_det;
            hits[i] = HitItem(
                pos, dir, nrm,
                ray.samples[i].wavelength,
                ray.samples[i].time + detDist / ray.samples[i].constants.vg,
                contrib
            );
            //update ray weight
            ray.samples[i].lin_contrib *= w_scatter;
        }

        //sanity check: did we actually hit the target after sampling it?
        //if not, do not create a hit, but keep tracing
        if (isinf(detDist))
            return false;
    }

    //process all hit items
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        if (hits[i].time <= params.maxTime)
            response(hits[i]);
    }

    //done -> return hit flag
    return hit;
}

Ray sampleRay(uint idx) {
    SourceRay source = sampleLight(idx);  

    //transform source to tracing ray
    Medium medium = Medium(params.medium);
    Sample samples[N_LAMBDA];
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        samples[i] = Sample(
            source.samples[i].wavelength,
            source.samples[i].startTime,
            source.samples[i].contrib,
            0.0,
            lookUpMedium(medium, source.samples[i].wavelength));
    }
    return Ray(
        source.position,
        source.direction,
        samples
    );    
}

void main() {
    uint idx = gl_GlobalInvocationID.x;  
    if (idx >= BATCH_SIZE)
        return;

    Ray ray = sampleRay(idx);
    //discard ray if inside target
    if (distance(ray.position, params.target.position) <= params.target.radius)
        return;
    
    //trace loop
    uint dim = DIM_OFFSET;
    [[unroll]] for (uint i = 0; i < N_SCATTER; ++i, dim += DIM_STRIDE) {
        //trace() returns true, if we should stop tracing
        if (trace(ray, idx, dim)) return;
    }
}
