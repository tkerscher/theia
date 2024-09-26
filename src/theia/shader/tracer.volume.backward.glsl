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

#include "ray.glsl"
#include "ray.combine.glsl"

#include "wavelengthsource.common.glsl"
#include "lightsource.common.glsl"
#include "response.common.glsl"
#include "camera.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "camera.glsl"
#include "light.glsl"
#include "source.glsl"
#include "response.glsl"

#include "callback.util.glsl"

#ifndef DISABLE_DIRECT_LIGHTING
#include "tracer.direct.common.glsl"
#endif

#ifndef DISABLE_SELF_SHADOWING
bool isVisible(vec3 observer, vec3 target) {
    vec3 dir = target - observer;
    float dist = length(dir);
    dir /= dist;

    //Check if we are shadowed by target
    float t = intersectSphere(params.target, observer, dir);
    return t <= dist;
}
#else
bool isVisible(vec3 observer, vec3 target) {
    return true;
}
#endif

void traceShadowRay(
    BackwardRay ray,
    const CameraHit cam,
    uint idx, inout uint dim
) {
    //sample light
    SourceRay source = sampleLight(
        ray.state.position, vec3(0.0),
        ray.state.wavelength,
        idx, dim);
    //check if light is visible
    if (!isVisible(source.position, ray.state.position))
        return;
    
    //create hit by combining source and camera ray
    HitItem hit;
    ResultCode result = combineRays(ray, source, cam, params.propagation, hit);
    if (result >= 0) {
        response(hit);
    }
}

ResultCode trace(
    inout BackwardRay ray,
    const CameraHit cam,
    uint idx, inout uint dim
) {
    //sample distance
    float u = random(idx, dim);
    float dist = sampleScatterLength(ray, params.propagation, u);

    //check for self shadowing
    #ifndef DISABLE_SELF_SHADOWING
    float t = intersectSphere(params.target, ray.state.position, ray.state.direction);
    bool hit = t <= dist;
    dist = min(t, dist);
    #else
    bool hit = false;
    #endif
    
    //update ray. Even if self shadowed for correct callback
    ResultCode result = propagateRay(ray, dist, params.propagation);
    updateRayIS(ray, dist, params.propagation, hit);

    //abort if self shadowed
    if (hit) return RESULT_CODE_RAY_ABSORBED;
    //abort if out of bounds
    if (result < 0) return result;

    //create shadow ray
    traceShadowRay(ray, cam, idx, dim);
    return RESULT_CODE_RAY_SCATTERED;
}

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= BATCH_SIZE)
        return;
    
    //Direct light sampling
    #ifndef DISABLE_DIRECT_LIGHTING
    sampleDirect(idx, dim, Medium(params.medium), params.propagation);
    uint iPath = 2;
    #else
    uint iPath = 0;
    #endif

    //sample camera ray
    WavelengthSample photon = sampleWavelength(idx, dim);
    CameraRay cam = sampleCameraRay(photon.wavelength, idx, dim);
    BackwardRay ray = createRay(cam, getMedium(), photon);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, iPath++);

    //trace loop
    //one iteration less than PATH_LENGTH as we will always create shadow rays
    //extending the path length by one
    [[unroll]] for (uint i = 1; i < PATH_LENGTH; ++i) {
        //trace ray
        ResultCode result = trace(ray, cam.hit, idx, dim);
        onEvent(ray, result, idx, iPath++);

        //stop codes are negative
        if (result < 0) return;

        //scatter ray to prepare next trace, except on last iteration
        if (i < PATH_LENGTH - 1) {
            scatterRay(ray, random2D(idx, dim));
        }
    }
}
