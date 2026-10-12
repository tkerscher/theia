#ifndef _INCLUDE_VOLUME_NEE_MIS_PROXY
#define _INCLUDE_VOLUME_NEE_MIS_PROXY

#ifdef INLINE_VOLUME_MODEL

//rename to free up the function name again
#define estimateInScattering estimateInScattering_imp
#include "tracer/scene/backward/volume.nee.glsl"
#undef estimateInScattering

void estimateInScattering(
    BackwardRay ray,
    const CameraHit camHit,
    float tMax,
    inout uint dim
) {
    //no in scattering in transparent media
    uint volIdx = getVolumeIdx(ray.mediumIdx);
    if (volIdx == TRANSPARENT_IDX) return;

    estimateInScattering_imp(ray, camHit, tMax, dim);
}

#else

struct VolumeNeeData {
    BackwardRay ray;
    float tMax;
    uint dim;
};
layout(location = 3) callableDataEXT VolumeNeeData volumeNeeData;

void estimateInScattering(
    BackwardRay ray,
    const CameraHit camHit,
    float tMax,
    inout uint dim
) {
    //no in scattering in transparent media
    uint volIdx = getVolumeIdx(ray.mediumIdx);
    if (volIdx == TRANSPARENT_IDX) return;

    volumeNeeData.ray = ray;
    volumeNeeData.tMax = tMax;
    volumeNeeData.dim = dim;

    executeCallableEXT(PROXY_STRIDE * (volIdx - 1) + 3, 3);

    dim += VOLUME_IN_SCATTERING_MIS_DIM_STRIDE;
}

#endif

#endif
