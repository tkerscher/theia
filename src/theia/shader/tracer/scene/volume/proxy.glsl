#ifndef _INCLUDE_VOLUME_PROXY
#define _INCLUDE_VOLUME_PROXY

#include "math.glsl"
#include "tracer/scene/volume/index.glsl"

//Need to specify what direction or ray to use
#ifndef PROXY_RAY
#error "No ray model was specified!"
#endif

//There are two cases we want to handle here:
// - Besides transparent (e.g. vacuum), there is at most a single other volume model:
//    -> we can inline the model and not pay the cost of callables
// - There are multiple volume models:
//    -> we have to dynamically execute callables

#ifdef INLINE_VOLUME_MODEL

//we need to rename the original function names to be able to inject a bit more logic
#define applyVolume applyVolume_model
#define applyVolumeSampled applyVolumeSampled_model
#define sampleInteractionLength sampleInteractionLength_model

#include "volume.glsl"

#undef applyVolume
#undef applyVolumeSampled
#undef sampleInteractionLength

//create small shims to match the proxy with the model functions

float sampleInteractionLength_proxy(
    const PROXY_RAY ray,
    uint volIdx,
    inout uint dim
) {
    return sampleInteractionLength_model(ray, gl_LaunchIDEXT.x, dim);
}

ResultCode applyVolumeSampled_proxy(
    inout PROXY_RAY ray,
    float dist,
    bool hit,
    uint volIdx,
    inout uint dim
) {
    return applyVolumeSampled_model(ray, dist, hit, gl_LaunchIDEXT.x, dim);
}

#ifndef RAY_PARTICLE

ResultCode applyVolume_proxy(
    inout PROXY_RAY ray,
    float dist,
    bool hit,
    uint volIdx,
    inout uint dim
) {
    return applyVolume_model(ray, dist, hit, gl_LaunchIDEXT.x, dim);
}

#endif

#else //#ifdef INLINE_VOLUME_MODEL

struct LengthData {
    PROXY_RAY ray;
    uint dim;
    float dist;
};
layout(location = 0) callableDataEXT LengthData lengthData;

struct ApplyData{
    PROXY_RAY ray;
    float dist;
    uint dim;
    ResultCode result;
};
layout(location = 1) callableDataEXT ApplyData applyData;

#ifdef RAY_PARTICLE
#define PROXY_STRIDE 2
#else
#define PROXY_STRIDE 3
#endif

float sampleInteractionLength_proxy(
    const PROXY_RAY ray, uint volIdx, inout uint dim
) {
    uint callableIdx = PROXY_STRIDE * (volIdx - 1) + 0;
    lengthData.ray = ray;
    lengthData.dim = dim;
    executeCallableEXT(callableIdx, 0);
    //read back results
    dim = lengthData.dim;
    return lengthData.dist;
}

ResultCode applyVolumeSampled_proxy(
    inout PROXY_RAY ray,
    float dist,
    bool hit,
    uint volIdx,
    inout uint dim
) {
    uint callableIdx = PROXY_STRIDE * (volIdx - 1) + 1;
    applyData.ray = ray;
    //put hit flag into sign of dist
    applyData.dist = hit ? -dist : dist;
    // applyData.dist = copySignBit(dist, hit ? -1.0 : 1.0);
    applyData.dim = dim;
    executeCallableEXT(callableIdx, 1);
    //read back result
    ray = applyData.ray;
    dim = applyData.dim;
    return applyData.result;
}

#ifndef RAY_PARTICLE

ResultCode applyVolume_proxy(
    inout PROXY_RAY ray,
    float dist,
    bool hit,
    uint volIdx,
    inout uint dim
) {
    uint callableIdx = PROXY_STRIDE * (volIdx - 1) + 2;
    applyData.ray = ray;
    //put hit flag into sign of dist
    applyData.dist = hit ? -dist : dist;
    // applyData.dist = copySignBit(dist, hit ? -1.0 : 1.0);
    applyData.dim = dim;
    executeCallableEXT(callableIdx, 1);
    //read back result
    ray = applyData.ray;
    dim = applyData.dim;
    return applyData.result;
}

#endif

#endif

float sampleInteractionLength(
    const PROXY_RAY ray,
    uint idx,
    inout uint dim
) {
    //always inline transparent case as it's quite trivial
    uint volIdx = getVolumeIdx(ray.mediumIdx);
    if (volIdx == TRANSPARENT_IDX)
        return 1.0 / 0.0; //+inf
    
    //otherwise delegate call to correct model via proxy
    return sampleInteractionLength_proxy(ray, volIdx, dim);
}

ResultCode applyVolumeSampled(
    inout PROXY_RAY ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
) {
    //always inline transparent case as it's quite trivial
    uint volIdx = getVolumeIdx(ray.mediumIdx);
    if (volIdx == TRANSPARENT_IDX)
        return RESULT_CODE_SUCCESS;

    //otherwise delegate call to correct model via proxy
    return applyVolumeSampled_proxy(ray, dist, hit, volIdx, dim);
}

#ifndef RAY_PARTICLE

ResultCode applyVolume(
    inout PROXY_RAY ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
) {
    //always inline transparent case as it's quite trivial
    uint volIdx = getVolumeIdx(ray.mediumIdx);
    if (volIdx == TRANSPARENT_IDX)
        return RESULT_CODE_SUCCESS;

    //otherwise delegate call to correct model via proxy
    return applyVolume_proxy(ray, dist, hit, volIdx, dim);
}

#endif

#endif
