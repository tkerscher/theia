layout(local_size_x = 512) in;

#include "result.glsl"
#include "material.glsl"

#include "ray.glsl"
#include "rng.glsl"
#include "photon.glsl"
#include "source.glsl" //either light source or camera
#include "volume.glsl"

uniform SamplerParams {
    uvec2 queueAdr;
    uint queueSize;

    float sampleCoef;
    float maxDist;
    float hitChance;
} params;

#ifdef SAMPLE_FORWARD

#define RAY ForwardRay
uint saveRay(const ForwardRay ray, uint pointer) {
    saveForwardRay(params.queueAdr, params.queueSize, pointer, ray);
    return pointer + RAY_FIELD_COUNT * params.queueSize;
}

#else

#define RAY BackwardRay
uint saveRay(const BackwardRay ray, uint pointer) {
    saveBackwardRay(params.queueAdr, params.queueSize, pointer, ray);
    return pointer + RAY_FIELD_COUNT * params.queueSize;
}

#endif

#ifdef RAY_PARTICLE

ResultCode _propagate(inout ForwardRay ray, bool hit, uint idx, inout uint dim) {
    float dist = sampleInteractionLength(ray, idx, dim);
    dist = min(dist, params.maxDist);
    #ifdef SAMPLER_PROPAGATE_RAY
    ResultCode code = propagateRay(ray, dist);
    if (code < 0) return code;
    #endif
    return applyVolumeSampled(ray, dist, hit, idx, dim);
}

#else

ResultCode _propagate(inout RAY ray, bool hit, uint idx, inout uint dim) {
    //sample step length
    float dist;
    if (params.sampleCoef > 0.0)
        dist = -log(1.0 - random(idx, dim)) / params.sampleCoef;
    else
        dist = sampleInteractionLength(ray, idx, dim);
    dist = min(dist, params.maxDist);
    
    //propagate ray if requested
    #ifdef SAMPLER_PROPAGATE_RAY
    ResultCode code = propagateRay(ray, dist);
    if (code < 0) return code;
    #endif

    //apply volume effects
    if (params.sampleCoef > 0.0)
        return applyVolume(ray, dist, hit, idx, dim);
    else
        return applyVolumeSampled(ray, dist, hit, idx, dim);
}

#endif

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

    //save input
    uint p = idx;
    p = saveRay(ray, p);

    //sample hit?
    bool hit = random(idx, dim) < params.hitChance;

    //propagate ray
    ResultCode code = _propagate(ray, hit, idx, dim);
    //sample volume interaction if applicable
    if (!hit && code >= 0) {
        code = sampleVolumeInteraction(ray, idx, dim);
    }

    //store results
    p = saveRay(ray, p);
    IntBuffer ints = IntBuffer(params.queueAdr);
    ints.values[p] = code; p += params.queueSize;
    ints.values[p] = hit ? 1 : 0;
}
