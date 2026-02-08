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

//instead of storing massive samples in the payload, we just store the RNG state
//and resample with the exact same state to produce the exact same samples
struct DirectRay {
    uint dim;
    ResultCode result;
};
layout(location = DIRECT_RAY_PAYLOAD_LOCATION) rayPayloadInEXT DirectRay directRay;

//we just assume that maxTime is the first element in TraceParams
//GLSL won't check on this, so one needs to be especially carefull!
//(It should not be part of the payload as it is uniform across all invocations)
uniform TraceParams {
    float maxTime;
} params;

ResultCode combine(
    ForwardRay ray,
    BackwardRay camRay,
    CameraHit camHit,
    uint idx, uint dim
) {
    //we already know both rays are mutually visible
    // -> propagate either one of them and combine
    ResultCode result;
    float dist = distance(ray.position, camRay.position);
    result = propagateRay(camRay, dist); if (result < 0) return result;
    result = applyVolume(camRay, dist, true, idx, dim); if (result < 0) return result;
    HitItem hit;
    result = combineRaysAligned(ray, camRay, camHit, hit); if (result < 0) return result;

    //check hit within time window
    #ifdef RAY_TRANSIENT
    if (hit.time > params.maxTime) return RESULT_CODE_RAY_DECAYED;
    #endif

    //finally, process hit
    response(hit, idx, dim);
    //success
    return RESULT_CODE_RAY_DETECTED;
}

void main() {
    //at this point we know the light and camera are mutually visible
    // -> recreate samples and create hit item

    //recreate RNG state
    uint idx = gl_LaunchIDEXT.x;
    uint dim = directRay.dim;
    //produce exact same samples by using same RNG state
    float lambdaContrib;
    float wavelength = sampleWavelength(lambdaContrib, idx, dim);
    CameraSample camSample = sampleCamera(wavelength, idx, dim);
    ForwardRay ray = sampleLight(
        camSample.position,
        camSample.normal,
        wavelength,
        camSample.mediumIdx,
        idx, dim
    );
    ray.lin_contrib *= lambdaContrib;
    //create corresponding backward ray
    CameraHit camHit;
    BackwardRay camRay = createCameraRay(camSample, ray.direction, wavelength, camHit);

    //combine rays
    directRay.result = combine(ray, camRay, camHit, idx, dim);

    //notify
    onEvent(ray, directRay.result, idx, 1);
}
