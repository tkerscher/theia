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
#define DIM_STRIDE 5

layout(local_size_x = BLOCK_SIZE) in;

#include "material.glsl"
#include "ray.propagate.glsl"
#include "ray.sample.glsl"
#include "scatter.surface.glsl"
#include "scene.intersect.glsl"
#include "sphere.glsl"
#include "tracer.mis.glsl"
#include "util.branch.glsl"

#include "lightsource.common.glsl"
#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "source.glsl"
#include "response.glsl"

uniform accelerationStructureEXT tlas;
layout(scalar) readonly buffer Detectors {
    Sphere detectors[];
};

layout(scalar) uniform TraceParams {
    uint targetIdx;
    uvec2 sceneMedium;

    PropagateParams propagation;
} params;

bool processHit(inout Ray ray, rayQueryEXT rayQuery, bool update) {
    //process ray query
    vec3 objPos, objNrm, objDir, worldPos, worldNrm, geomNormal;
    Material mat;
    bool inward;
    bool success = processRayQuery(
        ray, rayQuery,
        mat, inward,
        objPos, objNrm, objDir,
        worldPos, worldNrm,
        geomNormal
    );
    if (CHECK_BRANCH(!success))
        return true; //abort tracing

    //offset position to prevent self-intersection
    if (!inward)
        geomNormal = -geomNormal;
    worldPos = offsetRay(worldPos, geomNormal);
    //calculate distance
    float dist = length(worldPos - ray.position);

    //update samples
    success = updateSamples(ray, dist, params.propagation, false, true);
    if (CHECK_BRANCH(!success))
        return true;

    //calculate reflectance (needed for both branches)
    float r[N_LAMBDA];
    vec3 dir = normalize(ray.direction);
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        r[i] = reflectance(
            mat,
            ray.samples[i].wavelength,
            ray.samples[i].constants.n,
            dir, worldNrm
        );
    }

    //If hit target -> create response item
    int customId = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    if (customId == params.targetIdx && (mat.flags & MATERIAL_TARGET_BIT) != 0) {
        //process hits
        [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
            //skip out of bound samples
            if (ray.samples[i].time > params.propagation.maxTime)
                continue;

            Sample s = ray.samples[i];
            //attenuate by transmission
            float contrib = exp(s.log_contrib) * s.lin_contrib * (1.0 - r[i]);
            response(HitItem(
                objPos, objDir, objNrm,
                s.wavelength,
                s.time,
                contrib
            ));
        }
    }

    //update if needed
    if (CHECK_BRANCH(update)) {
        //geometric effect (Lambert's cosine law)
        float lambert = -dot(dir, worldNrm);
        [[unroll]] for (int i = 0; i < N_LAMBDA; ++i) {
            ray.samples[i].lin_contrib *= r[i] * lambert;
        }
        //update ray
        ray.position = worldPos;
        ray.direction = normalize(reflect(dir, worldNrm));
    }

    //done
    return false; //dont abort tracing
}

bool processScatter(
    inout Ray ray,
    float dist,
    rayQueryEXT rayQuery,
    uint idx, uint dim, // rng coords
    out Ray hitRay
) {
    //propagate to scatter event
    bool success = propagateRay(ray, dist, params.propagation, true, false);
    if (CHECK_BRANCH(!success))
        return true; //abort tracing

    //volume scatter event: MIS both phase function and detector
    Sphere detector = detectors[params.targetIdx];
    vec4 rng = vec4(random2D(idx, dim), random2D(idx, dim)); dim += 4;
    vec3 detDir, scatterDir;
    float w_det, w_scatter, detDist;
    scatterMIS(
        detector, Medium(ray.medium),
        ray.position, ray.direction,
        rng,
        detDir, w_det, detDist, scatterDir, w_scatter
    );

    //update ray; create copy for shadow ray
    hitRay = ray;
    ray.direction = scatterDir;
    hitRay.direction = detDir;
    //update samples
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        //udpate scattered ray
        ray.samples[i].lin_contrib *= w_scatter;
        //update detector ray
        hitRay.samples[i].lin_contrib *= w_det;
    }

    //trace shadow ray
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF,
        hitRay.position,
        0.0,
        detDir,
        detDist);
    rayQueryProceedEXT(rayQuery);

    //done
    return false; //dont abort trace
}

bool trace(inout Ray ray, uint idx, uint dim) {
    //just to be safe
    vec3 dir = normalize(ray.direction);
    //sample distance
    float u = random(idx, dim); dim++;
    float dist = -log(1.0 - u) / params.propagation.scatterCoefficient;

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
    bool hit = rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT;

    //volume scattering produces shadow rays aimed at the detector, which are
    //handled similar to direct hits. To reduce divergence, we handle them both
    //the same code.

    //create a local copy of ray, as it might either be the original one or
    //the shadow ray
    Ray hitRay;
    //check hit across subgroup, giving change to skip unnecessary code
    if (subgroupAll(!hit)) {
        //returns true, if we should abort tracing
        if (processScatter(ray, dist, rayQuery, idx, dim, hitRay))
            return true; //abort tracing
    }
    else if (subgroupAll(hit)) {
        hitRay = ray;
    }
    else {
        //mixed branching
        if (!hit) {
            //returns true, if we should abort tracing
            if (processScatter(ray, dist, rayQuery, idx, dim, hitRay))
                return true; //abort tracing
        }
        else {
            hitRay = ray;
        }
    }

    //handle hit: either directly from tracing or indirect via shadow ray
    //will also create a hit item in the queue (if one was created)
    bool hitResult = processHit(hitRay, rayQuery, hit);

    //copy hitRay back to ray if necessary
    if (subgroupAll(hit)) {
        //if direct hit -> update ray for tracing
        ray = hitRay;
        return hitResult;
    }
    else if (subgroupAll(!hit)) {
        //always continue after volume scattering
        return false; //dont abort tracing
    }
    else {
        //mixed branching
        if (hit) {
            ray = hitRay;
            return hitResult;
        }
        else {
            return false; //dont abort tracing
        }
    }
}

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= BATCH_SIZE)
        return;
    
    //sample ray
    Ray ray = createRay(sampleLight(idx), Medium(params.sceneMedium));

    //trace loop
    uint dim = DIM_OFFSET;
    [[unroll]] for (uint i = 0; i < N_SCATTER; ++i, dim += DIM_STRIDE) {
        //trace() returns true, if we should stop tracing
        if (trace(ray, idx, dim)) return;
    }
}
