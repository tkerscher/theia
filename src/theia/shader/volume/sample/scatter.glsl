layout(local_size_x = 512) in;

#include "result.glsl"
#include "material.glsl"
#include "util/sample.glsl"

#include "ray.glsl"
#include "rng.glsl"
#include "photon.glsl"
#include "source.glsl" //either light source or camera
#include "volume.glsl"

uniform SamplerParams {
    uvec2 queueAdr;
    uint queueSize;
    float propDist;
} params;

#ifdef SAMPLE_FORWARD
uint saveRay(const ForwardRay ray, uint pointer) {
    saveForwardRay(params.queueAdr, params.queueSize, pointer, ray);
    return pointer + RAY_FIELD_COUNT * params.queueSize;
}
#else
uint saveRay(const BackwardRay ray, uint pointer) {
    saveBackwardRay(params.queueAdr, params.queueSize, pointer, ray);
    return pointer + RAY_FIELD_COUNT * params.queueSize;
}
#endif

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint dim = 0;
    if (idx >= params.queueSize) return;

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

    //propagate if requested
    if (params.propDist >= 0.0) {
        applyVolume(ray, params.propDist, false, idx, dim);
    }

    //sample scattering
    float probSampled;
    vec3 newDir = sampleVolumeScattering(
        ray, probSampled, idx, dim
    );
    float probEval = volumeScatterProb(ray, newDir);
    //eval random dir
    vec3 randDir = sampleUnitSphere(random2D(idx, dim));
    float probRand = volumeScatterProb(ray, randDir);
    //scatter ray
    ResultCode result = volumeScatterRay(ray, randDir, idx, dim);

    //save results
    p = saveRay(ray, p);
    IntBuffer ints = IntBuffer(params.queueAdr);
    ints.values[p] = result; p += params.queueSize;
    FloatBuffer floats = FloatBuffer(params.queueAdr);
    floats.values[p] = newDir.x; p += params.queueSize;
    floats.values[p] = newDir.y; p += params.queueSize;
    floats.values[p] = newDir.z; p += params.queueSize;
    floats.values[p] = randDir.x; p += params.queueSize;
    floats.values[p] = randDir.y; p += params.queueSize;
    floats.values[p] = randDir.z; p += params.queueSize;
    floats.values[p] = probSampled; p += params.queueSize;
    floats.values[p] = probEval; p += params.queueSize;
    floats.values[p] = probRand; p += params.queueSize;
}
