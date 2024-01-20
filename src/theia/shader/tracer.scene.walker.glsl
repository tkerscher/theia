#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_ray_query : require
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
#ifndef DISABLE_MIS
#define DIM_STRIDE 7
#else
#define DIM_STRIDE 5
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "material.glsl"
#include "ray.propagate.glsl"
#include "ray.sample.glsl"
#include "result.glsl"
#include "sphere.glsl"
#include "tracer.mis.glsl"
#include "tracer.scene.hit.glsl"
#include "util.branch.glsl"

#include "lightsource.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "source.glsl"

uniform accelerationStructureEXT tlas;
#ifndef DISABLE_MIS
layout(scalar) readonly buffer Targets {
    Sphere targets[];
};
#endif

layout(scalar) uniform TraceParams {
    uint targetIdx;
    uvec2 sceneMedium;
    //probability to choose detector instead of phase function for next event sampling
    float pTarget;

    PropagateParams propagation;
} params;

ResultCode processScatter(
    inout Ray ray,
    float dist,
    rayQueryEXT rayQuery,
    uint idx, uint dim //rng coords
) {
    //propagate to scatter event
    ResultCode result = propagateRay(ray, dist, params.propagation, true, false);
    if (CHECK_BRANCH(result < 0))
        return result; //abort tracing

#ifndef DISABLE_MIS    
    //volume scatter event: MIS both phase function and target
    Sphere target = targets[params.targetIdx];
    vec3 detDir, scatterDir;
    float w_det, w_scatter, detDist;
    scatterMIS_balance(
        target, Medium(ray.medium),
        ray.position, ray.direction,
        random2D(idx, dim), random2D(idx, dim + 2),
        detDir, w_det, detDist, scatterDir, w_scatter
    );
    //advance rng
    dim += 4;

    //one-sample both proposed direction
    float u = random(idx, dim); dim++;
    float w;
    if (u < params.pTarget) { //strictly less, so 0.0 disables it
        //next event from target sample
        ray.direction = detDir;
        w = w_det / params.pTarget;
    }
    else {
        //next event follows scattering phase function
        ray.direction = scatterDir;
        w = w_scatter / (1.0 - params.pTarget);
    }
    //update samples
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        ray.samples[i].lin_contrib *= w;
    }
#else
    //sample new direction from phase function
    float pScatter;
    ray.direction = scatter(
        Medium(ray.medium),
        ray.direction,
        random2D(idx, dim),
        pScatter //not needed
    );
#endif

    //done
    return RESULT_CODE_RAY_SCATTERED; //dont abort trace
}

ResultCode trace(inout Ray ray, uint idx, uint dim) {
    //just to be safe
    vec3 dir = normalize(ray.direction);
    //sample distance
    float u = random(idx, dim); dim++;
    float dist = sampleScatterLength(ray, params.propagation, u);

    //trace ray
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF, //mask -> hit anything
        ray.position,
        0.0, //t_min; self-intersections are handled via offsets
        dir,
        dist);
    rayQueryProceedEXT(rayQuery);
    //check if we hit anything
    bool hit = rayQueryGetIntersectionTypeEXT(rayQuery, true) ==
        gl_RayQueryCommittedIntersectionTriangleEXT;

    //draw a random number for reflect/transmit decision
#ifndef DISABLE_TRANSMISSION
    //dont bother drawing a random number if we dont use it
    u = random(idx, dim); dim++;
#endif
    //advance rng
    dim++;

    //handle hit
    ResultCode result;
    if (CHECK_BRANCH(hit)) {
        result = processHit(
            ray, rayQuery, u,
            params.targetIdx,
            params.propagation,
            true
        );
    }
    else if (CHECK_BRANCH(!hit)) {
        result = processScatter(
            ray, dist, rayQuery,
            idx, dim
        );
    }

    //done
    return result;
}

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= BATCH_SIZE)
        return;
    
    //sample ray
    Ray ray = createRay(sampleLight(idx), Medium(params.sceneMedium));
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);

    //trace loop
    uint dim = DIM_OFFSET;
    [[unroll]] for (uint i = 1; i <= N_SCATTER; ++i, dim += DIM_STRIDE) {
        ResultCode result = trace(ray, idx, dim);
        //user provided callback
        onEvent(ray, result, idx, i);
        //stop codes are negative
        if (result < 0) return;
    }
}
