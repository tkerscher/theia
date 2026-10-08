layout(local_size_x = 512) in;

#include "result.glsl"
#include "material.glsl"
#include "util/sample.glsl"

#include "ray.glsl"
#include "rng.glsl"
#include "photon.glsl"
#include "source.glsl"
#include "volume.glsl"

uniform SamplerParams {
    uvec2 queueAdr;
    uint queueSize;
    float maxStep;
} samplerParams;

#ifdef SAMPLE_FORWARD
uint saveRay(const ForwardRay ray, uint pointer) {
    saveForwardRay(samplerParams.queueAdr, samplerParams.queueSize, pointer, ray);
    return pointer + RAY_FIELD_COUNT * samplerParams.queueSize;
}
#else
uint saveRay(const BackwardRay ray, uint pointer) {
    saveBackwardRay(samplerParams.queueAdr, samplerParams.queueSize, pointer, ray);
    return pointer + RAY_FIELD_COUNT * samplerParams.queueSize;
}
#endif

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint dim = 0;
    if (idx >= samplerParams.queueSize) return;

    #ifdef SAMPLE_FORWARD
    ForwardRay ray = sampleLight(idx, dim);
    #else
    float lambdaContrib;
    float lambda = sampleWavelength(lambdaContrib, idx, dim);
    CameraHit camHit;
    BackwardRay ray = sampleCameraRay(lambda, camHit, idx, dim);
    #endif

    //save input
    uint p = idx;
    p = saveRay(ray, p);

    //sample nee step
    float sampledProb;
    float sampledStep = sampleVolumeNeeStepSize(
        ray, samplerParams.maxStep, sampledProb, idx, dim);
    //eval
    float evalStep = random(idx, dim) * samplerParams.maxStep;
    float evalProb = evalVolumeNeeStepSize(ray, samplerParams.maxStep, evalStep);

    //save results
    FloatBuffer floats = FloatBuffer(samplerParams.queueAdr);
    floats.values[p] = sampledStep; p += samplerParams.queueSize;
    floats.values[p] = sampledProb; p += samplerParams.queueSize;
    floats.values[p] = evalStep; p += samplerParams.queueSize;
    floats.values[p] = evalProb; p += samplerParams.queueSize;
}
