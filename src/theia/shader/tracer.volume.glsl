#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require

//Check expected macros
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef PATH_LENGTH
#error "PATH_LENGTH not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "ray.propagate.glsl"
#include "sphere.intersect.glsl"

layout(scalar) uniform TraceParams {
    Sphere target;
    uvec2 medium;

    PropagateParams propagation;
} params;

//define global medium
#define USE_GLOBAL_MEDIUM
Medium getMedium() {
    return Medium(params.medium);
}
#include "ray.medium.glsl"

#include "ray.response.glsl"
#include "ray.scatter.glsl"
#include "result.glsl"
#include "tracer.mis.glsl"

#include "wavelengthsource.common.glsl"
#include "lightsource.common.glsl"
#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "source.glsl"
#include "photon.glsl"
#include "response.glsl"

#include "callback.util.glsl"

void createResponse(ForwardRay ray, vec3 dir, float weight, bool scattered) {
    //try to hit target: dist == inf -> no hit
    float dist = intersectSphere(params.target, ray.state.position, dir);
    if (isinf(dist))
        return; // missed target -> no response
    
    //scatter ray if needed
    if (scattered) {
        scatterRayIS(ray, dir);
    }
    
    //calculate hit coordinates
    vec3 hitPos = ray.state.position + dir * dist;
    vec3 hitNormal = normalize(hitPos - params.target.position);
    vec3 objPos = hitNormal * params.target.radius;
    //propagate ray to hit
    ResultCode code = propagateRayToHit(ray, hitPos, hitNormal, params.propagation);
    if (code < 0) return; //out-of-bounds

    //create (weighted) response
    ray.state.lin_contrib *= weight;
    response(createHit(ray, objPos, hitNormal));
}

ResultCode trace(
    inout ForwardRay ray,
    uint idx, inout uint dim,
    bool first, bool allowResponse
) {
    //sample distance
    float u = random(idx, dim);
    float dist = sampleScatterLength(ray, params.propagation, u);

    //trace sphere
    float t = intersectSphere(params.target, ray.state.position, ray.state.direction);
    //check if we hit
    bool hit = t <= dist;

    //update ray
    dist = min(t, dist);
    ResultCode result = propagateRay(ray, dist, params.propagation);
    updateRayIS(ray, dist, params.propagation, hit);
    //check if ray is still in bounds
    if (result < 0)
        return result; //abort tracing

    //if we hit the sphere we assume the ray got absorbed -> abort tracing
    //we won't create a hit, as we already created a path of that length
    //in the previous step via MIS.
    //Only exception: first trace step, as we didn't MIS from the light source
    // can be disabled to use a direct light integrator instead
    // (partition of the path space)
    if (hit && allowResponse) {
        //calculate local coordinates (dir is the same)
        vec3 hitNormal = normalize(ray.state.position - params.target.position);
        vec3 hitObjPos = hitNormal * params.target.radius;
        //Align hit, i.e. rotate polRef to plane of incidence
        alignRayToHit(ray, hitNormal);
        response(createHit(ray, hitObjPos, hitNormal));

        return RESULT_CODE_RAY_DETECTED;
    }
    else if (hit) {
        //hit target, but response for this length was already sampled
        //-> abort without creating a response
        return RESULT_CODE_RAY_ABSORBED;
    }

    #ifndef DISABLE_MIS
    //MIS detector:
    //We both sample the scattering phase function as well as the detector
    //sphere for a possible hit direction
    float wTarget, wPhase;
    vec3 dirTarget, dirPhase;
    vec2 uTarget = random2D(idx, dim), uPhase = random2D(idx, dim);
    sampleTargetMIS(
        Medium(params.medium),
        ray.state.position, ray.state.direction, params.target,
        uTarget, uPhase,
        wTarget, dirTarget,
        wPhase, dirPhase
    );
    //create hits
    createResponse(ray, dirTarget, wTarget, true);
    createResponse(ray, dirPhase, wPhase, true);
    #endif

    //no hit -> scatter
    return RESULT_CODE_RAY_SCATTERED;
}

//process macro flags
#if defined(DISABLE_MIS) && !defined(DISABLE_DIRECT_LIGHTING)
#define DIRECT_LIGHTING true
#else
#define DIRECT_LIGHTING false
#endif

#ifndef DISABLE_MIS
#define ALLOW_RESPONSE false
#else
#define ALLOW_RESPONSE true
#endif

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= BATCH_SIZE)
        return;

    //sample ray
    Medium medium = Medium(params.medium);
    WavelengthSample photon = sampleWavelength(idx, dim);
    ForwardRay ray = createRay(
        sampleLight(photon.wavelength, idx, dim),
        medium, photon);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);
    //discard ray if inside target
    if (distance(ray.state.position, params.target.position) <= params.target.radius) {
        onEvent(ray, ERROR_CODE_TRACE_ABORT, idx, 0);
        return;
    }

    //try to extend first ray to sphere if direct lighting and MIS is enabled
    #if !defined(DISABLE_DIRECT_LIGHTING) && !defined(DISABLE_MIS)
    createResponse(ray, ray.state.direction, 1.0, false);
    #endif

    //trace loop: first iteration
    // special as it is allowed to create a direct hit (there was no MIS yet)
    ResultCode result = trace(ray, idx, dim, true, DIRECT_LIGHTING);
    onEvent(ray, result, idx, 1);
    if (result < 0)
        return;
    
    //trace loop: rest
    [[unroll]] for (uint i = 0; i < PATH_LENGTH; ++i) {
        //scatter ray
        vec2 u = random2D(idx, dim);
        float cos_theta, phi;
        scatterRay(ray, u);

        //trace ray
        ResultCode result = trace(ray, idx, dim, false, ALLOW_RESPONSE);
        onEvent(ray, result, idx, i + 2);

        //stop codes are negative
        if (result < 0)
            return;
    }
}
