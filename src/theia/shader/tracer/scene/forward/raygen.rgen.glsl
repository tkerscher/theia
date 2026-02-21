#include "result.glsl"
#include "material.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "callback.glsl"
#include "photon.glsl"
#include "source.glsl"

#include "tracer/scene/volume/proxy.glsl"
#include "tracer/propagate/forward.glsl"

#include "tracer/scene/forward/io.glsl"

#ifndef DISABLE_NEE
#include "tracer/scene/forward/nee.glsl"
#endif

layout(location = 0) rayPayloadEXT TraceData traceData;

ResultCode trace(
    inout ForwardRay ray,
    inout uint dim
) {
    //sample distance to propagate in volume
    float dist = sampleStepSize(ray, params.propagation, gl_LaunchIDEXT.x, dim);

    //setup trace payload
    traceData.ray = ray;
    traceData.dim = dim;
    //trace
    traceRayEXT(
        accelerationStructureEXT(params.tlas),
        gl_RayFlagsOpaqueEXT,
        0xFF,                                   //cull mask
        0,                                      //sbt offset
        0,                                      //sbt stride
        getVolumeIdx(ray.mediumIdx) + 1,        //miss index
        ray.position,                           //origin
        0.0,                                    //t_min
        ray.direction,                          //direction
        dist,                                   //t_max
        0                                       //payload location
    );
    //read back result
    ray = traceData.ray;
    // dim = traceData.dim;
    dim += TRACE_RNG_STRIDE; //save a bit of memory bandwidth
    return traceData.result;
}

void main() {
    uint dim = 0;
    uint idx = gl_LaunchIDEXT.x;
    #ifndef DISPATCH_INDIRECT
    //we could not limit the ray count on the host side so we do it here
    if (idx >= params.batchSize) return;
    #endif

    //sample light ray
    ForwardRay ray = sampleLight(idx, dim);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);

    //Direct light sampling by extending ray to light source
    #if !defined(DISABLE_DIRECT_LIGHTING) && !defined(DISABLE_NEE)
    traceNEE(ray, params.tlas, dim);
    #endif

    //trace loop
    for (uint i = 1; i <= PATH_LENGTH; ++i) {
        //trace ray
        ResultCode result = trace(ray, dim);

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
