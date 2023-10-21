#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require

#include "scatter.volume.glsl"
#include "sphere.glsl"
#include "wavefront.items.glsl"

layout(local_size_x = 32) in;

layout(scalar) readonly buffer VolumeScatterQueue{
    uint volumeCount;
    VolumeScatterItem volumeItems[];
};
layout(scalar) writeonly buffer RayQueue{
    uint rayCount;
    RayItem rayItems[];
};
layout(scalar) writeonly buffer ShadowQueue{
    uint shadowCount;
    ShadowRayItem shadowItems[];
};

layout(scalar) readonly buffer RNGBuffer{ float u[]; };
layout(scalar) readonly buffer Detectors{
    Sphere detectors[];
};

layout(scalar) uniform TraceParams{
    // for transient rendering, we won't importance sample the media
    float scatterCoefficient;

    float maxTime;
    vec3 lowerBBoxCorner;
    vec3 upperBBoxCorner;
} params;

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= volumeCount)
        return;
    VolumeScatterItem item = volumeItems[idx];
    Ray ray = item.ray;

    //calc new position
    ray.position = ray.position + ray.direction * item.dist;
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
        ray.photons[i].T_log += lambda * item.dist - mu_e * item.dist;
        ray.photons[i].T_lin /= lambda;

        //scattereing is mu_s*p(theta) -> mu_s shared by all processes
        float mu_s = ray.photons[i].constants.mu_s;
        ray.photons[i].T_lin *= mu_s;

        //update traveltime
        ray.photons[i].travelTime += item.dist / ray.photons[i].constants.vg;
        //bounds check
        if (ray.photons[i].travelTime <= params.maxTime)
            anyBelowMaxTime = true;
    }
    
    //bounds check: max time
    if (!anyBelowMaxTime)
        return;
    
    //sample target
    Sphere detector = detectors[item.targetIdx];
    vec2 rng = vec2(u[ray.rngIdx++], u[ray.rngIdx++]);
    float pTarget, targetDist;
    vec3 targetDir = sampleSphere(detector, ray.position, rng, targetDist, pTarget);
    Ray targetRay = ray; //copy
    //keep original direction for now, we'll need it for the weights
    //targetRay.direction = targetDir;

    //scatter
    rng = vec2(u[ray.rngIdx++], u[ray.rngIdx++]);
    float pScatter;
    vec3 scatterDir = scatter(ray, rng, pScatter);

    //calculate cross probs
    float pSc_target = sampleSphereProb(detector, ray.position, scatterDir);
    float pT_scatter = scatterProb(targetRay, targetDir);
    //calculate MIS weights (power heuristic)
    float w_scatter = pScatter*pScatter / (pScatter*pScatter + pSc_target*pSc_target);
    float w_target = pTarget*pT_scatter / (pTarget*pTarget + pT_scatter*pT_scatter); //(!)
    //Now it's safe to update direction
    targetRay.direction = targetDir;

    //update photons
    for (int i = 0; i < N_PHOTONS; ++i) {
        //scatter coefficient
        float mu_s = ray.photons[i].constants.mu_s;

        //update scatter sampled
        ray.photons[i].T_lin *= w_scatter * mu_s;
        //update target sampled
        targetRay.photons[i].T_lin *= w_target * mu_s;
    }

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
    rayItems[oldCount + id] = RayItem(ray, item.targetIdx);

    //Now do the same with the shadow rays
    if (subgroupElect()) {
        oldCount = atomicAdd(shadowCount, n);
    }
    oldCount = subgroupBroadcastFirst(oldCount);
    shadowItems[oldCount + id] = ShadowRayItem(targetRay, item.targetIdx, targetDist);
}
