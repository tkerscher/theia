#include "result.glsl"
#include "material.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "callback.glsl"
#include "photon.glsl"
#include "source.glsl"

#include "tracer/scene/target/config.glsl"
#include "tracer/scene/volume/proxy.glsl"
#include "tracer/scene/target/propagate.glsl"

#include "tracer/scene/target/io.glsl"

#ifdef WAVEFRONT_COMPACTION
#include "tracer/scene/target/wave.glsl"
#endif

#ifndef DISABLE_NEE
#include "tracer/scene/target/nee.glsl"
#endif

layout(location = 0) rayPayloadEXT TraceData traceData;

ResultCode trace(
    inout TRACE_RAY ray,
    uint idx,
    inout uint dim
) {
    //sample distance to propagate in volume
    float dist = sampleStepSize(ray, params.propagation, idx, dim);

    //setup trace payload
    traceData.ray = ray;
    traceData.idx = idx;
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
        gl_RayFlagsOpaqueEXT,                   //ray flags
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
    #else
    //trace without reorder
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
    #endif
    //read back result
    ray = traceData.ray;
    // dim = traceData.dim;
    dim += TRACE_RNG_STRIDE; //save a bit of memory bandwidth
    return traceData.result;
}

void main() {
    uint dim = 0;
    uint idx = gl_LaunchIDEXT.x;
    //number of iterations this path has already completed in earlier waves
    uint pathLength = 0;
    TRACE_RAY ray;

#ifdef WAVEFRONT_COMPACTION
    /*******************************************************************
    *   Wavefront compaction: only the first wave samples the source,   *
    *   the later ones pick the paths that survived out of the queue.   *
    *******************************************************************/
    if (waveParams.firstWave == 0) {
        //gl_LaunchIDEXT.x is the queue slot here, NOT the path id: that one is
        //restored from the record.
        loadWaveRay(
            waveParams.srcQueue, waveParams.capacity, gl_LaunchIDEXT.x,
            ray, idx, dim, pathLength
        );
    }
    else
#endif
    {
        #ifndef DISPATCH_INDIRECT
        //we could not limit the ray count on the host side so we do it here
        if (idx >= params.batchSize) return;
        #endif

        //sample source
        #if TRACE_DIRECTION ==  TRACE_DIRECTION_FORWARD
        //sample light ray
        ray = sampleLight(idx, dim);
        onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);
        #elif TRACE_DIRECTION == TRACE_DIRECTION_BACKWARD
        //sample camera
        float lambdaContrib;
        float lambda = sampleWavelength(lambdaContrib, idx, dim);
        CameraHit camHit;
        ray = sampleCameraRay(lambda, camHit, idx, dim);
        ray.lin_contrib *= lambdaContrib;
        onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);
        #else
        #error "Invalid trace direction"
        #endif

        //Direct light sampling by extending ray to light source
        #if !defined(DISABLE_DIRECT_LIGHTING) && !defined(DISABLE_NEE)
        traceNEE(ray, params.tlas, idx, dim);
        #endif
    }

    //trace loop
#ifdef WAVEFRONT_COMPACTION
    uint last = pathLength + waveParams.iterations;
#else
    uint last = PATH_LENGTH;
#endif
    for (uint i = pathLength + 1; i <= last; ++i) {
        //trace ray
        ResultCode result = trace(ray, idx, dim);

        //change result if we are on the last iteration
        if (result >= 0 && i >= PATH_LENGTH - 1) {
            result = RESULT_CODE_MAX_ITER;
        }
        //notify
        onEvent(ray, result, idx, i);

        //stop codes are negative
        if (result < 0) return;
    }

#ifdef WAVEFRONT_COMPACTION
    /*******************************************************************
    *   Still alive after this wave -> hand the path over to the next   *
    *   one. The counter both allocates the slots and sizes the next    *
    *   (indirect) dispatch.                                            *
    *                                                                   *
    *   One atomic per subgroup instead of one per lane: it is 32x      *
    *   fewer atomics, and it keeps the survivors of a subgroup in      *
    *   CONSECUTIVE slots.                                              *
    *******************************************************************/
    uint count = subgroupAdd(1u);
    uint offset = subgroupExclusiveAdd(1u);
    uint base = 0;
    if (subgroupElect()) {
        base = atomicAdd(Counter(waveParams.dstCounter).count, count);
    }
    base = subgroupBroadcastFirst(base);
    saveWaveRay(
        waveParams.dstQueue, waveParams.capacity, base + offset,
        ray, idx, dim, last
    );
#endif
}
