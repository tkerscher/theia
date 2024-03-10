#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_scalar_block_layout : require
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
#ifndef PATH_LENGTH
#error "PATH_LENGTH not defined"
#endif
#ifndef DIM_OFFSET
#error "DIM_OFFSET not defined"
#endif
// #samples per iteration
#ifndef DISABLE_MIS
#define DIM_STRIDE 8
#else
#define DIM_STRIDE 4
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "ray.propagate.glsl"
#include "ray.sample.glsl"
#include "scene.intersect.glsl"
#include "scene.traverse.glsl"
#include "sphere.intersect.glsl"
#include "tracer.mis.glsl"

#include "lightsource.common.glsl"
#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "source.glsl"
#include "response.glsl"

uniform accelerationStructureEXT tlas;
#ifndef DISABLE_MIS
layout(scalar) readonly buffer Targets {
    Sphere targets[];
};
#endif

layout(scalar) uniform TraceParams {
    uint targetIdx;
    uvec2 sceneMedium;

    PropagateParams propagation;
} params;

void traceShadowRay(Ray ray, vec3 dir, float dist, float weight) {
    //trace ray against scene
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF, //mask -> hit everything
        ray.position,
        0.0, dir, dist);
    rayQueryProceedEXT(rayQuery);

    //check if we hit anything
    bool hit = rayQueryGetIntersectionTypeEXT(rayQuery, true) ==
        gl_RayQueryCommittedIntersectionTriangleEXT;
    if (!hit) return;
    //process hit
    vec3 objPos, objNrm, objDir, worldPos, worldNrm, geomNormal;
    Material mat;
    bool inward;
    ResultCode result = processRayQuery(
        ray, rayQuery,
        mat, inward,
        objPos, objNrm, objDir,
        worldPos, worldNrm,
        geomNormal
    );
    dist = distance(worldPos, ray.position);
    uint flags = inward ? mat.flagsInwards : mat.flagsOutwards;
    if (result < 0) return;

    //check if we hit the target
    int customId = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    if (customId != params.targetIdx || (flags & MATERIAL_DETECTOR_BIT) != 0)
        return;
    
    //propagate ray
    result = propagateRay(ray, dist, params.propagation, false);
    if (result < 0) return;

    //create response
    bool black = (flags & MATERIAL_BLACK_BODY_BIT) != 0;
    createResponse(
        ray, mat,
        params.propagation.maxTime,
        objPos, objDir, objNrm, worldNrm,
        black, weight
    );
}

void processScatter(inout Ray ray, uint idx, uint dim) {
    //apply scatter coefficient
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        ray.samples[i].lin_contrib *= ray.samples[i].constants.mu_s;
    }

#ifndef DISABLE_MIS
    //MIS detector:
    //we both sample the scattering phase function as well as the detector
    //sphere for a possible hit direction
    float wTarget, wPhase;
    vec3 dirTarget, dirPhase;
    vec2 uTarget = random2D(idx, dim), uPhase = random2D(idx, dim + 2); dim += 4;
    Sphere target = targets[params.targetIdx];
    sampleTargetMIS(
        Medium(ray.medium),
        ray.position, ray.direction, target,
        uTarget, uPhase,
        wTarget, dirTarget,
        wPhase, dirPhase
    );
    //check if phase sampling even has a chance of hitting the detector
    //TODO: check if this test actually reduce pressure on ray tracing hardware
    //      otherwise this only introduce divergence overhead
    bool phaseMiss = isinf(intersectSphere(target, ray.position, dirPhase));
    //trace shadow ray & create hit if visible
    //(dist is max distance checked for hit: dist to back of target sphere)
    float dist = distance(ray.position, target.position) + target.radius;
    traceShadowRay(ray, dirTarget, dist, wTarget);
    if (!phaseMiss)
        traceShadowRay(ray, dirPhase, dist, wPhase);
#endif

    //sample phase function for new ray direction
    float _ignore;
    vec2 u = random2D(idx, dim); dim += 2;
    ray.direction = scatter(Medium(ray.medium), ray.direction, u, _ignore);
}

ResultCode trace(inout Ray ray, uint idx, uint dim, bool allowResponse, bool last) {
    //sample distance
    float u = random(idx, dim); dim++;
    float dist = sampleScatterLength(ray, params.propagation, u);

    //trace ray against scene
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF, //mask -> hit everything
        ray.position,
        0.0, //t_min; self-intersections handled via offsets
        normalize(ray.direction),
        dist);
    rayQueryProceedEXT(rayQuery);
    //check if we hit anything
    bool hit = rayQueryGetIntersectionTypeEXT(rayQuery, true) ==
        gl_RayQueryCommittedIntersectionTriangleEXT;
    
    //fetch actual travelled distance if we hit anything
    if (hit) {
        dist = rayQueryGetIntersectionTEXT(rayQuery, true);
    }
    //propagate ray (ray, dist, params, scatter)
    ResultCode result = propagateRay(ray, dist, params.propagation, false);
    updateRayIS(ray, dist, params.propagation, hit);
    //check if propagation was sucessfull
    if (result < 0)
        return result; //abort tracing
    
    //handle either intersection or scattering
    if (hit) {
        u = random(idx, dim); dim++;
        result = processHit(
            ray, rayQuery,
            params.targetIdx,
            params.propagation.maxTime,
            u, allowResponse
        );
    }
    else {
        //dont bother scattering on the last iteration (we wont hit anything)
        //this also prevents MIS to sample paths one longer than PATH_LENGTH
        if (!last)
            processScatter(ray, idx, dim + 1);
        result = RESULT_CODE_RAY_SCATTERED;
    }

    //done
    return result;
}

//Toggle direct lighting
#ifndef DISABLE_DIRECT_LIGHTING
#define ALLOW_RESPONSE_INIT true
#else
#define ALLOW_RESPONSE_INIT false
#endif

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= BATCH_SIZE)
        return;
    
    //sample ray
    Medium medium = Medium(params.sceneMedium);
    Ray ray = createRay(sampleLight(idx), medium);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);
    //advange rng by amount used by sampleLight()
    uint dim = DIM_OFFSET;

    //trace loop
    bool allowResponse = ALLOW_RESPONSE_INIT;
    [[unroll]] for (uint i = 1; i <= PATH_LENGTH; ++i, dim += DIM_STRIDE) {
        //trace ray
        bool last = i == PATH_LENGTH; //mark last trace
        ResultCode result = trace(ray, idx, dim, allowResponse, last);
        onEvent(ray, result, idx, i);
        //stop codes are negative
        if (result < 0)
            return;
        
#ifndef DISABLE_MIS
        //we did MIS on target if we scattered in volume
        // -> disable response for next segment
        allowResponse = result != RESULT_CODE_RAY_SCATTERED;
#else
        allowResponse = true;
#endif
    }
}
