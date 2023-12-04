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

/******************************* MACRO SETTINGS *******************************/
// BATCH_SIZE:      int, work group size
// HIT_QUEUE_SIZE:  int, max #items in hit queue
// N_PHOTONS:       int, #photons per ray
// N_SCATTER:       int, #scatter iterations to run
// OUTPUT_RAYS:     def, if defined, saves rays after tracing
// QUEUE_SIZE:      int, max #items in ray queue

#include "hits.glsl"
#include "scatter.volume.glsl"
#include "sphere.intersect.glsl"
#include "ray.glsl"
#include "rng.glsl"

layout(local_size_x = BATCH_SIZE) in;

layout(scalar) readonly buffer RayQueueBuffer {
    uint rayCount;
    RayQueue rayQueue;
};
layout(scalar) writeonly buffer HitQueueBuffer {
    uint hitCount;
    HitQueue hitQueue;
};
#ifdef OUTPUT_RAYS
layout(scalar) writeonly buffer OutRayQueueBuffer {
    uint outRayCount;
    RayQueue outRayQueue;
};

layout(push_constant) uniform PushConstant {
    int saveRays; // bool flag 
};
#endif

layout(scalar) uniform TraceParams {
    Sphere target;
    float scatterCoefficient;

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

    //update all photons
    float lambda = params.scatterCoefficient;
    bool anyBelowMaxTime = false;
    [[unroll]] for (uint i = 0; i < N_PHOTONS; ++i) {
        float mu_e = ray.photons[i].constants.mu_e;
        ray.photons[i].log_contrib += lambda * dist - mu_e * dist;
        ray.photons[i].time += dist / ray.photons[i].constants.vg;
        if (!hit) ray.photons[i].lin_contrib != lambda;
        //time boundary check
        if (ray.photons[i].time <= params.maxTime)
            anyBelowMaxTime = true;
    }
    //return result of boundary check
    return anyBelowMaxTime;
}

//traces ray
//returns true, if target was hit and tracing should be stopped
bool trace(inout Ray ray) {
    //sample distance
    float u = random(ray.rngStream, ray.rngCount++);
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
    PhotonHit hits[N_PHOTONS];
    
    if (hit) {
        //create normal
        nrm = normalize(ray.position - params.target.position);
        //transform pos to sphere local coords / recalc to improve float error
        pos = nrm * params.target.radius;
        [[unroll]] for (uint i = 0; i < N_PHOTONS; ++i) {
            hits[i] = PhotonHit(
                ray.photons[i].wavelength,
                ray.photons[i].time,
                exp(ray.photons[i].log_contrib) * ray.photons[i].lin_contrib
            );
        }
    }
    else {
        //MIS both volume scatter and detector hit
        //We'll use the following naming scheme: pXY, where
        //X: sampled distribution
        //Y: prob, distribution
        //S: scatter, D: detector
        //e.g. pSD: p_det(dir ~ scatter)
        
        //sample target
        vec2 rng = random2D(ray.rngStream, ray.rngCount);
        ray.rngCount += 2;
        float pDD, targetDist;
        vec3 targetDir = sampleSphere(params.target, ray.position, rng, targetDist, pDD);
        targetDist = intersectSphere(params.target, ray.position, targetDir);
        //create hit params
        //in the rare case, that we dit not hit the target: targetDist = +inf
        //it's easier programming wise to check for this later and ignore the
        //corresponding hit. Due to branching there should not be any performance hit
        pos = ray.position + targetDir * targetDist;
        dir = targetDir;
        nrm = normalize(pos - params.target.position);
        //transform pos to sphere local coords / recalc to improve float error
        pos = nrm * params.target.radius;

        //scatter
        Medium medium = Medium(ray.medium);
        rng = random2D(ray.rngStream, ray.rngCount);
        ray.rngCount += 2;
        float pSS;
        vec3 scatterDir = scatter(medium, ray.direction, rng, pSS);

        //calculate cross probs pSD, pDS
        float pSD = scatterProb(medium, ray.direction, targetDir);
        float pDS = sampleSphereProb(params.target, ray.position, scatterDir);

        //calculate MIS weights
        float w_scatter = pSS*pSS / (pSS*pSS + pSD*pSD);
        float w_target = pDD / (pDD*pDD + pDS*pDS);
        //^^^ Note that we already canceled out the extra pDD from the estimator
        // -> f(x)/pDD * w_target

        //update ray
        ray.direction = scatterDir;
        //iterate photons to update and create hits
        [[unroll]] for (uint i = 0; i < N_PHOTONS; ++i) {
            ray.photons[i].lin_contrib *= ray.photons[i].constants.mu_s;
            //create hits
            float mu_e = ray.photons[i].constants.mu_e;
            float contrib = exp(ray.photons[i].log_contrib - mu_e * dist);
            contrib *= ray.photons[i].lin_contrib * w_target;
            hits[i] = PhotonHit(
                ray.photons[i].wavelength,
                ray.photons[i].time + targetDist / ray.photons[i].constants.vg,
                contrib
            );
            //update ray weight
            ray.photons[i].lin_contrib *= w_scatter;
        }

        //sanity check: did we actually hit the target after sampling it?
        //if not, do not create a hit, but keep tracing
        if (isinf(targetDist))
            return false;
    }

    //save hits to queue
    uint n = subgroupAdd(1);
    uint oldCount = 0;
    if (subgroupElect()) {
        oldCount = atomicAdd(hitCount, n);
    }
    oldCount = subgroupBroadcastFirst(oldCount);
    uint id = subgroupExclusiveAdd(1);
    uint idx = oldCount + id;
    SAVE_HIT(pos, dir, nrm, hits, hitQueue, idx)

    //done -> return hit flag
    return hit;
}

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= rayCount)
        return;
    //load ray
    LOAD_RAY(ray, rayQueue, idx)
    
    //discard ray if inside target
    if (distance(ray.position, params.target.position) <= params.target.radius)
        return;    

    //trace loop
    [[unroll]] for (uint i = 0; i < N_SCATTER; ++i) {
        //trace() returns true, if we should stop tracing
        if (trace(ray)) return;
    }

    //optionally: save ray state for further tracing
#ifdef OUTPUT_RAYS
    //check if we should save rays in this batch
    //(may be skipped on last round)
    if (saveRays != 0) {
        uint n = subgroupAdd(1);
        uint oldCount = 0;
        if (subgroupElect()) {
            oldCount = atomicAdd(outRayCount, n);
        }
        oldCount = subgroupBroadcastFirst(oldCount);
        uint id = subgroupExclusiveAdd(1);
        idx = oldCount + id;
        SAVE_RAY(ray, outRayQueue, idx)
    }
#endif
}
