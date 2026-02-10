#ifndef _INCLUDE_TRACER_SCENE_DIRECT_SAMPLING
#define _INCLUDE_TRACER_SCENE_DIRECT_SAMPLING

#include "tracer/scene/volume/index.glsl"

//instead of storing massive samples in the payload, we just store the RNG state
//and resample with the exact same state to produce the exact same samples
struct DirectRay {
    uint dim;
    ResultCode result;
};
layout(location = DIRECT_RAY_PAYLOAD_LOCATION) rayPayloadEXT DirectRay directRay;

void sampleDirect(uvec2 tlas, inout uint dim) {
    //save rng state
    uint idx = gl_LaunchIDEXT.x;
    uint rngDimInit = dim;
    directRay.dim = dim;

    //sample ray
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
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);

    //light might be on the wrong side of the camera. In that case we do not
    //need to even trace a shadow ray
    if (dot(camSample.normal, ray.direction) >= 0.0) {
        onEvent(ray, RESULT_CODE_RAY_MISSED, idx, 1);
        return;
    }

    //in the miss shader additionally to resampling we also propagate the ray
    //and produce a response, potentially requiring more rng draws.
    //Instead of reading back the rng state, we just advance it deterministically
    //by a known amount and save us from an unnecessary round trip through memory
    dim = rngDimInit + DIRECT_SAMPLING_RNG_STRIDE;

    //since we just want to do a simply visibility test it does not matter in
    //what direction we trace. However, we expect the camera side potentially
    //requiring more precision as there might be complex geometry
    //-> trace from camera to light
    vec3 position = camSample.position;
    vec3 direction = normalize(ray.position - position);
    float dist = distance(ray.position, position);

    //we will use the direct sampling routine from other tracers too
    //requiring some configurability
    uint volIdx = getVolumeIdx(camSample.mediumIdx);
    uint missIdx = DIRECT_SAMPLING_MISS_SHADER_STRIDE * volIdx + DIRECT_SAMPLING_MISS_SHADER_OFFSET;

    //Will remain if the no-op hit shader is called
    directRay.result = RESULT_CODE_RAY_ABSORBED;
    //trace shadow ray to check for visibility
    traceRayEXT(
        accelerationStructureEXT(tlas),
        gl_RayFlagsTerminateOnFirstHitEXT,
        0xFF,                               //cull mask
        DIRECT_SAMPLING_HIT_SHADER_OFFSET,  //sbt offset
        0,                                  //sbt stride
        missIdx,                            //miss index
        position,                           //origin
        0.0,                                //t_min
        direction,                          //direction
        dist,                               //t_max
        DIRECT_RAY_PAYLOAD_LOCATION         //payload location
    );
    //read back result code and notify if nothing happens
    if (directRay.result == RESULT_CODE_RAY_ABSORBED) {
        onEvent(ray, RESULT_CODE_RAY_ABSORBED, idx, 1);
        // return;
    }

    //at this point we are done. If light and camera are mutually visible, the
    //miss shader will get invoked which in turn will handle the creation of a
    //hit item and calling the response function on it. Otherwise, we provided
    //a no-op hit shader and nothing happens.
}

#endif
