#include "result.glsl"
#include "material.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "callback.glsl"
#include "camera.glsl"
#include "photon.glsl"

#include "tracer/scene/volume/proxy.glsl"
#include "tracer/propagate/backward.glsl"

#ifndef DISABLE_DIRECT_LIGHTING

//direct sampler also needs the light source
#include "source.glsl"
//configure direct sampler
#define DIRECT_SAMPLING_MISS_SHADER_OFFSET 1
#define DIRECT_SAMPLING_MISS_SHADER_STRIDE MISS_STRIDE
#define DIRECT_SAMPLING_HIT_SHADER_OFFSET 1
#include "tracer/scene/direct/sample.glsl"

#endif

uniform TraceParams {
    uvec2 tlas;
    PropagationParams propagation;
    uint batchSize;
} params;

struct TraceData{
    BackwardRay ray;
    uint dim;
    ResultCode result;
};
layout(location = 0) rayPayloadEXT TraceData traceData;

#ifndef DISABLE_VOLUME_SCATTER_MIS

#include "tracer/scene/backward/volume.nee.proxy.glsl"

ResultCode trace(inout BackwardRay ray, const CameraHit camHit, uint dim) {
    //sample distance to propagate in volume
    float dist = sampleStepSize(ray, params.propagation, gl_LaunchIDEXT.x, dim);
    //upper limit we can sample for  volume scattering NEE
    float tMax = maxStepSize(ray, params.propagation);
    //edge case: tMax <= 0.0, meaning the ray is already lost
    //(guards against undefined behavior in the trace call)
    if (tMax <= 0.0) return RESULT_CODE_RAY_LOST;

    //trace ray with tMax so we know the limit for NEE
    uint missIdx = MISS_STRIDE * getVolumeIdx(ray.mediumIdx);
    hitObjectEXT hitObj;
    hitObjectTraceRayEXT(
        hitObj,
        accelerationStructureEXT(params.tlas),
        gl_RayFlagsOpaqueEXT,
        0xFF,
        0,
        0,
        missIdx,
        ray.position,
        0.0,
        ray.direction,
        tMax,
        0
    );
    //at this point no miss or hit shader has been executed yet

    //update tMax for inscattering sampling
    //TODO: Thanks to finite precision, at tMax we might be already in the
    //      intersected surface. If NEE samples that point, the shadow ray fails.
    //      Ideally, one would choose a tMax that ensure this won't happen.
    //TODO: It would be nice to only execute the callable shader if there is a
    //      a real chance for NEE to succeed, i.e. volume supports scattering
    tMax = hitObjectGetRayTMaxEXT(hitObj);
    //estimate volume in scattering via callable
    if (tMax > 0.0) //there can be no in-scattering for zero length
        estimateInScattering(ray, camHit, tMax, dim);

    //we traced with tMax instead of the sampled step size -> correct the state
    if (!hitObjectIsHitEXT(hitObj) || tMax > dist) {
        hitObjectRecordMissEXT(
            hitObj,
            gl_RayFlagsOpaqueEXT,   //flags
            missIdx,                //miss index
            ray.position,           //origin
            0.0,                    //t_min
            ray.direction,          //direction
            dist                    //t_max
        );
    }
    //finally, optionally reorder and execute miss or hit shader
    traceData.ray = ray;
    traceData.dim = dim;
    #ifdef TRACER_REORDER_THREADS
    hitObjectReorderExecuteEXT(hitObj, 0);
    #else
    hitObjectExecuteShaderEXT(hitObj, 0);
    #endif
    //read back result
    ray = traceData.ray;
    return traceData.result;
}

#else

ResultCode trace(inout BackwardRay ray, const CameraHit camHit, uint dim) {
    //sample distance to propagate in volume
    float dist = sampleStepSize(ray, params.propagation, gl_LaunchIDEXT.x, dim);

    //setup trace payload
    traceData.ray = ray;
    traceData.dim = dim;
    //trace and optionally, reorder
    #ifdef TRACER_REORDER_THREADS
    //in theory one could also give special hints for reordering or aborting
    //tracing before executing miss and hit shaders, but the effect on performance
    //was minimal, so we skip that for now
    hitObjectEXT hitObj;
    hitObjectTraceReorderExecuteEXT(
        hitObj,
        accelerationStructureEXT(params.tlas),
        gl_RayFlagsOpaqueEXT,                       //ray flags
        0xFF,                                       //cull mask
        0,                                          //sbt offset
        0,                                          //sbt stride
        MISS_STRIDE * getVolumeIdx(ray.mediumIdx),  //miss index
        ray.position,                               //origin
        0.0,                                        //t_min
        ray.direction,                              //direction
        dist,                                       //t_max
        0                                           //payload location
    );
    #else
    //trace without reorder
    traceRayEXT(
        accelerationStructureEXT(params.tlas),
        gl_RayFlagsOpaqueEXT,
        0xFF,                                       //cull mask
        0,                                          //sbt offset
        0,                                          //sbt stride
        MISS_STRIDE * getVolumeIdx(ray.mediumIdx),  //miss index
        ray.position,                               //origin
        0.0,                                        //t_min
        ray.direction,                              //direction
        dist,                                       //t_max
        0                                           //payload location
    );
    #endif
    //read back result
    ray = traceData.ray;
    // dim = traceData.dim;
    return traceData.result;
}

#endif

void main() {
    uint dim = 0;
    uint idx = gl_LaunchIDEXT.x;
    #ifndef DISPATCH_INDIRECT
    //we could not limit the ray count on the host side so we do it here
    if (idx >= params.batchSize) return;
    #endif

    //Direct light sampling
    #ifndef DISABLE_DIRECT_LIGHTING
    sampleDirect(params.tlas, dim);
    #endif

    //sample camera
    dim = CAMERA_SAMPLE_RNG_DIM;
    float lambdaContrib;
    float lambda = sampleWavelength(lambdaContrib, idx, dim);
    CameraHit camHit;
    BackwardRay ray = sampleCameraRay(lambda, camHit, idx, dim);
    ray.lin_contrib *= lambdaContrib;
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);

    //trace loop
    for (uint i = 1; i < PATH_LENGTH; ++i) {
        //trace ray
        ResultCode result = trace(ray, camHit, dim);
        //advance rng        
        dim += TRACE_RNG_STRIDE;

        //change result if we are on the last iteration
        if (result >= 0 && i >= PATH_LENGTH - 1) {
            result = RESULT_CODE_MAX_ITER;
        }
        //notify
        onEvent(ray, result, idx, i);

        //stop codes are negative
        if (result < 0) return;
    }
}
