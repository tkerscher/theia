#ifndef _INCLUDE_VOLUME_TRACER_DIRECT_LIGHTING
#define _INCLUDE_VOLUME_TRACER_DIRECT_LIGHTING

#include "tracer/volume/shadowing.glsl"

ResultCode combine(
    inout ForwardRay ray,
    CameraSample camSample,
    float maxTime,
    uint idx, inout uint dim
) {
    ResultCode result;
    //create camera ray
    CameraHit camHit;
    BackwardRay camRay = createCameraRay(
        camSample, ray.direction, ray.wavelength, camHit
    );

    //propagate rays towards each other
    float dist = distance(ray.position, camRay.position);
    result = propagateRay(ray, dist); if (result < 0) return result;
    result = applyVolume(camRay, dist, true, idx, dim); if (result < 0) return result;
    HitItem hit;
    result = combineRaysAligned(ray, camRay, camHit, hit); if (result < 0) return result;

    //check if hit is within time window
    #ifdef RAY_TRANSIENT
    if (hit.time > maxTime) return RESULT_CODE_RAY_DECAYED;
    #endif

    //success -> process hit
    response(hit, idx, dim);
    return RESULT_CODE_RAY_DETECTED;
}

void sampleDirect(
    float maxTime,
    uint idx, inout uint dim
) {
    //sample light ray pointing at camera
    float lambdaContrib;
    float lambda = sampleWavelength(lambdaContrib, idx, dim);
    CameraSample camSample = sampleCamera(lambda, idx, dim);
    ForwardRay ray = sampleLight(
        camSample.position,
        camSample.normal,
        lambda,
        camSample.mediumIdx,
        idx, dim
    );
    ray.lin_contrib *= lambdaContrib;
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);

    ResultCode result;
    //discard sample if light is on the wrong side of the camera
    if (dot(camSample.normal, ray.direction) >= 0.0) {
        result = RESULT_CODE_RAY_MISSED;
    }
    //optionally, check for self-shadowing
    #ifndef DISABLE_SELF_SHADOWING
    else if (!isVisible(ray.position, camSample.position)) {
        result = RESULT_CODE_RAY_MISSED;
    }
    #endif
    //everything is fine -> combine light source with camera
    else {
        result = combine(ray, camSample, maxTime, idx, dim);       
    }

    //notify
    onEvent(ray, result, idx, 1);
}

#endif
