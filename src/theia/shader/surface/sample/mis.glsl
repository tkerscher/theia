layout(local_size_x = 512) in;

#include "result.glsl"
#include "material.glsl"
#include "scene/types.glsl"
#include "util/sample.glsl"

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

    //sample scattering (MIS)
    float probSampled;
    vec3 newDir = sampleSurfaceScattering(
        ray, hit,
        #ifdef SurfaceProperties
        props,
        #endif
        probSampled,
        idx, dim
    );
    float probEval = surfaceScatterProb(
        ray, hit,
        #ifdef SurfaceProperties
        props,
        #endif
        newDir
    );
    //eval random dir
    vec3 randDir = sampleHemisphere(random2D(idx, dim));
    float probRand = surfaceScatterProb(
        ray, hit,
        #ifdef SurfaceProperties
        props,
        #endif
        randDir
    );
    ResultCode result = surfaceScatterRay(
        ray, hit,
        #ifdef SurfaceProperties
        props,
        #endif
        randDir,
        idx, dim
    );

    //save results
    #ifdef SAMPLE_FORWARD
    saveForwardRay(params.queueAdr, params.queueSize, p, ray);
    #else
    saveBackwardRay(params.queueAdr, params.queueSize, p, ray);
    #endif
    p += RAY_FIELD_COUNT * params.queueSize;

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
