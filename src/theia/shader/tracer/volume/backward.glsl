layout(local_size_x = 512) in;

#include "result.glsl"
#include "material.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "callback.glsl"
#include "photon.glsl"
#include "source.glsl"
#include "camera.glsl"
#include "response.glsl"
#include "volume.glsl"

#include "tracer/propagate/backward.glsl"

uniform TraceParams {
    uint batchSize;
    
    PropagationParams propagation;
} params;

#ifndef DISABLE_SELF_SHADOWING

#include "target/common.glsl"
#include "target.glsl"

bool isVisible(vec3 observer, vec3 target) {
    vec3 dir = normalize(target - observer);
    float dist = distance(target, observer);

    //Check if we are shadowed by target
    // -> returns false (target not visible) if shadowed
    TargetSample hit = intersectTarget(observer, dir);
    return !hit.valid || (hit.dist >= dist);
}

#else

bool isVisible(vec3 observer, vec3 target) {
    return true;
}

#endif

#ifndef DISABLE_DIRECT_LIGHTING

void sampleDirect(
    float wavelength,
    float wavelengthContrib,
    uint idx, inout uint dim
) {
    //sample camera
    CameraHit camHit;
    CameraSample camSample = sampleCamera(wavelength, idx, dim);
    ForwardRay lightRay = sampleLight(
        camSample.position,
        camSample.normal,
        wavelength,
        camSample.mediumIdx,
        idx, dim
    );
    BackwardRay ray = createCameraRay(
        camSample,
        lightRay.direction,
        wavelength,
        camHit
    );
    ray.lin_contrib *= wavelengthContrib;
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);

    //combine rays and create hit
    HitItem hit;
    ResultCode result = combineRays(
        ray, lightRay, camHit,
        params.propagation,
        hit, idx, dim);
    if (result >= 0) {
        response(hit, idx, dim);
        result = RESULT_CODE_RAY_DETECTED;
    }
    onEvent(ray, result, idx, 1);
}

#endif

void traceShadowRay(
    BackwardRay ray,
    const CameraHit cam,
    uint idx, inout uint dim
) {
    //sample light
    ForwardRay source = sampleLight(
        ray.position, vec3(0.0),
        ray.wavelength,
        ray.mediumIdx,
        idx, dim
    );

    #ifndef DISABLE_SELF_SHADOWING
    //check if light is visible
    if (!isVisible(source.position, ray.position))
        return;
    #endif
    
    //create hit by combining light and camera ray
    HitItem hit;
    ResultCode result = combineRays(
        ray, source, cam,
        params.propagation,
        hit, idx, dim);
    if (result >= 0) {
        response(hit, idx, dim);
    }
}

ResultCode trace(
    inout BackwardRay ray,
    const CameraHit cam,
    uint idx, inout uint dim
) {
    //sample distance to propagate in volume
    float dist = sampleStepSize(ray, params.propagation, idx, dim);

    //check for self-shadowing
    #ifndef DISABLE_SELF_SHADOWING
    TargetSample intersection = intersectTarget(ray.position, ray.direction);
    bool hit = intersection.valid && intersection.dist <= dist;
    if (hit) dist = intersection.dist;
    #else
    bool hit = false;
    #endif

    //propagate ray even if self-shadowed for correct result code
    ResultCode result = propagate(
        ray, dist,
        hit, true,
        params.propagation,
        idx, dim);
    if (hit) return RESULT_CODE_RAY_ABSORBED;
    if (result < 0) return result;

    //trace shadow ray for NEE
    traceShadowRay(ray, cam, idx, dim);
    return RESULT_CODE_SUCCESS;
}

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.batchSize) return;

    //sample wavelength
    float lambdaContrib;
    float lambda = sampleWavelength(lambdaContrib, idx, dim);

    //optionally, sample direct light contribution
    #ifndef DISABLE_DIRECT_LIGHTING
    sampleDirect(lambda, lambdaContrib, idx, dim);
    #endif

    //sample camera ray
    CameraHit hit;
    BackwardRay ray = sampleCameraRay(lambda, hit, idx, dim);
    ray.lin_contrib *= lambdaContrib;
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);

    //trace loop
    for (uint i = 1; i < PATH_LENGTH; ++i) {
        //trace ray
        ResultCode result = trace(ray, hit, idx, dim);
        //scatter ray except on last iteration
        if (result >= 0) {
            if (i < PATH_LENGTH - 1) {
                result = sampleVolumeInteraction(ray, idx, dim);
            }
            else {
                result = RESULT_CODE_MAX_ITER;
            }
        }
        onEvent(ray, result, idx, i);

        //stop codes are negative
        if (result < 0) return;
    }
}
