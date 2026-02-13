#include "result.glsl"
#include "material.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "callback.glsl"
#include "camera.glsl"
#include "photon.glsl"

#include "tracer/scene/volume/proxy.backward.glsl"
#include "tracer/propagate/backward.glsl"

#ifndef DISABLE_DIRECT_LIGHTING

//direct sampler also needs the light source
#include "source.glsl"
//configure direct sampler
#define DIRECT_SAMPLING_MISS_SHADER_OFFSET 2
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

ResultCode trace(inout BackwardRay ray, uint dim) {
    //sample distance to propagate in volume
    float dist = sampleStepSize(ray, params.propagation, gl_LaunchIDEXT.x, dim);

    //setup trace payload
    traceData.ray = ray;
    traceData.dim = dim;
    //trace
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
    //read back result
    ray = traceData.ray;
    // dim = traceData.dim;
    return traceData.result;
}

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
    float lambdaContrib;
    float lambda = sampleWavelength(lambdaContrib, idx, dim);
    dim = CAMERA_SAMPLE_RNG_DIM;
    CameraHit camHit;
    BackwardRay ray = sampleCameraRay(lambda, camHit, idx, dim);
    ray.lin_contrib *= lambdaContrib;
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);

    //trace loop
    for (uint i = 1; i < PATH_LENGTH; ++i) {
        //trace ray
        ResultCode result = trace(ray, dim);
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
