#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require

#include "rng.glsl"
#include "scatter.volume.glsl"
#include "sphere.glsl"
#include "wavefront.common.glsl"

layout(local_size_x = LOCAL_SIZE) in;

layout(scalar) readonly buffer VolumeScatterQueueBuffer {
    uint volumeCount;
    VolumeScatterQueue volumeQueue;
};
layout(scalar) writeonly buffer RayQueueBuffer {
    uint rayCount;
    RayQueue rayQueue;
};
layout(scalar) writeonly buffer ShadowQueueBuffer {
    uint shadowCount;
    ShadowRayQueue shadowQueue;
};

layout(scalar) readonly buffer Detectors {
    Sphere detectors[];
};

layout(scalar) uniform Params {
    TraceParams params;
};

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= volumeCount)
        return;
    //load ray
    LOAD_RAY(ray, volumeQueue.rays, idx)
    float dist = volumeQueue.dist[idx];

    //calc new position
    ray.position = ray.position + ray.direction * dist;
    //boundary check
    if (any(lessThan(ray.position, params.lowerBBoxCorner)) ||
        any(greaterThan(ray.position, params.upperBBoxCorner)))
    {
        return;
    }

    //update photons
    float lambda = params.scatterCoefficient;
    bool anyBelowMaxTime = false;
    for (int i = 0; i < N_PHOTONS; ++i) {
        //update throughput, i.e. transmission/probability, for all wavelengths

        //prob travel distance is lambda*exp(-lambda*dist)
        //  -> we split the linear from the exp part for the update
        //attenuation is exp(-mu_e*dist)
        //  -> log(delta) = (mu_e-lambda)*dist
        float mu_e = ray.photons[i].constants.mu_e;
        //write out multplication in hope to prevent catastrophic cancelation
        ray.photons[i].log_contrib += lambda * dist - mu_e * dist;
        ray.photons[i].lin_contrib /= lambda;

        //scattereing is mu_s*p(theta) -> mu_s shared by all processes
        float mu_s = ray.photons[i].constants.mu_s;
        ray.photons[i].lin_contrib *= mu_s;

        //update traveltime
        ray.photons[i].time += dist / ray.photons[i].constants.vg;
        //bounds check
        if (ray.photons[i].time <= params.maxTime)
            anyBelowMaxTime = true;
    }
    
    //bounds check: max time
    if (!anyBelowMaxTime)
        return;
    
    //We'll use the following naming scheme: pXY, where
    //X: sampled distribution
    //Y: prob, distribution
    //S: scatter, D: detector
    //e.g. pSD: p_det(dir ~ scatter)

    //sample target
    Sphere detector = detectors[params.targetIdx];
    vec2 rng = random2D(ray.rngStream, ray.rngCount);
    ray.rngCount += 2;
    float pDD, targetDist;
    vec3 targetDir = sampleSphere(detector, ray.position, rng, targetDist, pDD);
    //Copy ray
    Ray targetRay = ray;

    //scatter
    Medium = Medium(ray.medium);
    rng = random2D(ray.rngStream, ray.rngCount);
    ray.rngCount += 2;
    float pSS;
    vec3 scatterDir = scatter(medium, ray.direction, rng, pSS);
    ray.direction = scatterDir;

    //calculate cross probs pSD, pDS
    float pSD = scatterProb(medium, targetRay.direction, targetDir);
    float pDS = sampleSphereProb(detector, ray.position, scatterDir);
    //Note that pSD is also the phase function

    //calculate MIS weights (power heuristic)
    float w_scatter = pSS*pSS / (pSS*pSS + pSD*pSD);
    float w_target = pDD / (pDD*pDD + pDS*pDS);
    //^^^ Note that we already canceled out the extra pDD from the estimator
    // -> f(x)/pDD * w_target

    //update photons
    for (int i = 0; i < N_PHOTONS; ++i) {
        //update scattered sample
        ray.photons[i].lin_contrib *= w_scatter;

        //update target sample
        targetRay.photons[i].lin_contrib *= pSD * w_target;
    }
    //update target direction
    targetRay.direction = targetDir;

    //count how many items we will be adding in this subgroup
    uint n = subgroupAdd(1);
    //elect one invocation to update counter in queue
    uint oldCount = 0;
    if (subgroupElect()) {
        oldCount = atomicAdd(rayCount, n);
    }
    //only the first(i.e. elected) one has correct oldCount value -> broadcast
    oldCount = subgroupBroadcastFirst(oldCount);
    //no we effectevily reserved us the range [oldCount..oldcount+n-1]

    //order the active invocations so each can write at their own spot
    uint id = subgroupExclusiveAdd(1);
    //create and save item
    idx = oldCount + id;
    SAVE_RAY(ray, rayQueue, idx)

    //Now do the same with the shadow rays
    if (subgroupElect()) {
        oldCount = atomicAdd(shadowCount, n);
    }
    oldCount = subgroupBroadcastFirst(oldCount);
    idx = oldCount + id;
    SAVE_RAY(targetRay, shadowQueue.rays, idx)
    shadowQueue.dist[idx] = targetDist;
}
