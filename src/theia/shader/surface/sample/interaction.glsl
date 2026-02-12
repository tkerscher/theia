layout(local_size_x = 512) in;

#include "result.glsl"
#include "material.glsl"
#include "scene/types.glsl"

#include "ray.glsl"
#include "rng.glsl"
#include "photon.glsl"
#include "source.glsl" //either light source or camera
#include "surface.glsl"

uniform SamplerParams {
    uvec2 queueAdr;
    uint queueSize;

    uint materialIdx;
    int objectId;

    vec3 surfaceNormal;

    vec3 offset;
    mat3x3 worldToObj;
} params;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint dim = 0;
    if (idx >= params.queueSize) return;

    #ifdef SAMPLE_FORWARD
    //sample ray from light source
    ForwardRay ray = sampleLight(idx, dim);
    #else
    //sample camera
    float lambdaContrib;
    float lambda = sampleWavelength(lambdaContrib, idx, dim);
    CameraHit camHit;
    BackwardRay ray = sampleCameraRay(lambda, camHit, idx, dim);
    #endif

    //fetch mediumIdx and flags
    uint mediumIdx, flags;
    vec3 surfaceNormal = normalize(params.surfaceNormal);
    bool inward = dot(ray.direction, surfaceNormal) < 0.0;
    queryMaterialSide(params.materialIdx, inward, mediumIdx, flags);
    //create surface hit
    SurfaceHit hit = SurfaceHit(
        params.materialIdx,
        flags,
        mediumIdx,
        inward,
        ray.position,
        inward ? surfaceNormal : -surfaceNormal,
        params.worldToObj * ray.position + params.offset,
        normalize(surfaceNormal * transpose(params.worldToObj)),
        params.worldToObj * ray.direction,
        params.worldToObj
    );

    //save inputs to queue
    uint p = idx;
    #ifdef SAMPLE_FORWARD
    saveForwardRay(params.queueAdr, params.queueSize, p, ray);
    #else
    saveBackwardRay(params.queueAdr, params.queueSize, p, ray);
    #endif
    p += RAY_FIELD_COUNT * params.queueSize;

    //optional preparation
    #ifdef SurfaceProperties
    SurfaceProperties props = prepareSurface(ray, hit, idx, dim);
    #endif
    //optionally, process target hit
    #ifdef SAMPLE_TARGET_HIT
    HitItem item;
    bool hitSuccess = processSurfaceTargetHit(
        ray,
        hit,
        #ifdef SurfaceProperties
        props,
        #endif
        params.objectId,
        item,
        idx, dim
    );
    #endif
    //query surface interaction
    ResultCode result = sampleSurfaceInteraction(
        ray,
        hit,
        #ifdef SurfaceProperties
        props,
        #endif
        idx, dim
    );

    //store results
    #ifdef SAMPLE_FORWARD
    saveForwardRay(params.queueAdr, params.queueSize, p, ray);
    #else
    saveBackwardRay(params.queueAdr, params.queueSize, p, ray);
    #endif
    p += RAY_FIELD_COUNT * params.queueSize;
    UIntBuffer uints = UIntBuffer(params.queueAdr);
    uints.values[p] = result; p += params.queueSize;

    #ifdef SAMPLE_TARGET_HIT
    uints.values[p] = uint(hitSuccess); p += params.queueSize;
    saveHitItem(params.queueAdr, params.queueSize, p, item);
    #endif
}
