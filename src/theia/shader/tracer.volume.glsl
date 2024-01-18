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
//alter ray layout
#define USE_GLOBAL_MEDIUM

layout(local_size_x = BLOCK_SIZE) in;

#include "material.glsl"
#include "sphere.intersect.glsl"
#include "ray.propagate.glsl"
#include "ray.sample.glsl"
#include "tracer.mis.glsl"
#include "util.branch.glsl"

#include "lightsource.common.glsl"
#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "source.glsl"
#include "response.glsl"

layout(scalar) uniform TraceParams {
    Sphere target;
    uvec2 medium;

    PropagateParams propagation;
} params;

//traces ray
//returns true, if target was hit and tracing should be stopped
bool trace(inout Ray ray, uint idx, uint dim) {
    //sample distance
    float u = random(idx, dim); dim++;
    float dist = sampleScatterLength(ray, params.propagation, u);

    //trace sphere
    vec3 dir = normalize(ray.direction); //just to be safe
    float t = intersectSphere(params.target, ray.position, dir);

    //check if we hit
    bool hit = t <= dist;
    //update ray
    bool good = propagateRay(ray, hit ? t : dist, params.propagation, false, hit);
    //check if trace is still in bounds
    if (CHECK_BRANCH(!good))
        return true; //abort tracing

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
        //volume scatter event: MIS both phase function and detector
        vec3 detDir, scatterDir;
        float w_det, w_scatter, detDist;
        scatterMIS_power(
            params.target, Medium(params.medium),
            ray.position, ray.direction,
            random2D(idx, dim), random2D(idx, dim + 2),
            detDir, w_det, detDist, scatterDir, w_scatter
        );
        //advance rng
        dim += 4;

        //hit detector
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
        if (hits[i].time <= params.propagation.maxTime)
            response(hits[i]);
    }

    //done -> return hit flag
    return hit;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;  
    if (idx >= BATCH_SIZE)
        return;

    Ray ray = createRay(sampleLight(idx), Medium(params.medium));
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
