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
#include "tracer/propagate/combine.glsl"
#include "tracer/volume/shadowing.glsl"
#ifndef DISABLE_DIRECT_LIGHTING
#include "tracer/volume/direct.glsl"
#endif

uniform TraceParams {
    uint batchSize;
    
    PropagationParams propagation;
} params;

#ifndef DISABLE_VOLUME_SCATTER_MIS

void traceShadowRay(
    BackwardRay ray,
    float tStep,
    const CameraHit cam,
    float pVolume,
    #ifdef LightNeeContext
    LightNeeContext lightNeeContext,
    #endif
    uint idx, inout uint dim
) {
    //propagate to scatter point
    //no visibility check needed as it's ensured by limiting the sampling earlier
    ResultCode result = propagate(
        ray, tStep, false, false, params.propagation, idx, dim);
    if (result < 0) return;

    //sample light source
    ForwardRay source = sampleLightMIS(
        ray.position, ray.direction, tStep,
        ray.wavelength, ray.mediumIdx,
        pVolume,
        #ifdef LightNeeContext
        lightNeeContext,
        #endif
        idx, dim
    );

    //optionally, check if light is visible from sampled scatter point
    #ifndef DISABLE_SELF_SHADOWING
    if (!isVisible(source.position, ray.position))
        return;
    #endif

    //create hit by combining light and camera ray
    HitItem hit;
    result = combineRays(
        ray, source, cam,
        params.propagation,
        hit, idx, dim);
    if (result >= 0) {
        response(hit, idx, dim);
    }
}

//we estimate the in scattered light by combining two sampling strategies via MIS:
//One defined by the light source itself, which allows e.g. point light sources
//to eliminate the weak singularity stemming from the 1/r^2 factor via equiangular
//sampling. The other is defined by the volume model and can consider e.g. the
//attenuation.
//Both return a step length in [0, tMax] where first an alternative volume scatter
//point is created which is then connected to the light source via a shadow ray.
void estimateInScattering(
    BackwardRay ray,
    const CameraHit cam,
    float tMax,
    uint idx, inout uint dim
) {
    //we use the balance heuristic with MIS. This turns the MIS weight normalization
    //into a plain sum, making it easier to calculate them across API boundaries
    //this is necessary to allow lights to cancel factors they want to get rid
    //of (like 1/r^2 in point light sources) in their contribution while still
    //applying the missing factor to the volume pdf to get a correct estimate.

    //query light source for a step size
    #ifdef LightNeeContext
    LightNeeContext lightNeeContext;
    #endif
    float tLight = sampleLightNeeStepSize(
        ray.position, ray.direction, tMax,
        ray.wavelength, ray.mediumIdx,
        #ifdef LightNeeContext
        lightNeeContext,
        #endif
        idx, dim
    );
    //evaluate volume model pdf needed for MIS weight
    float pVolume = evalVolumeNeeStepSize(ray, tMax, tLight);
    //trace shadow ray
    traceShadowRay(
        ray,
        tLight,
        cam,
        pVolume,
        #ifdef LightNeeContext
        lightNeeContext,
        #endif
        idx, dim
    );

    //query volume model for a step size
    float tVolume = sampleVolumeNeeStepSize(ray, tMax, pVolume, idx, dim);
    //evaluation of light pdf is handled by light source itself
    //-> can directly trace shadow ray
    traceShadowRay(
        ray,
        tVolume,
        cam,
        pVolume,
        #ifdef LightNeeContext
        lightNeeContext,
        #endif
        idx, dim
    );
}

ResultCode trace(
    inout BackwardRay ray,
    const CameraHit cam,
    uint idx, inout uint dim
) {
    //sample distance to propagate in volume
    float dist = sampleStepSize(ray, params.propagation, idx, dim);
    //upper limit we can sample for volume scattering NEE
    float tMax = maxStepSize(ray, params.propagation);

    //check for self-shadowing
    #ifndef DISABLE_SELF_SHADOWING
    TargetSample intersection = intersectTarget(ray.position, ray.direction);
    bool hit = intersection.valid && intersection.dist <= dist;
    if (hit) dist = intersection.dist;
    //limit volume NEE by hit if valid
    if (intersection.valid) tMax = min(tMax, intersection.dist);
    #else
    bool hit = false;
    #endif

    //estimate volume in scattering
    estimateInScattering(ray, cam, tMax, idx, dim);

    //propagate ray even if self-shadowed for correct ray state
    ResultCode result = propagateSampled(
        ray, dist,
        hit,
        params.propagation,
        idx, dim);
    return hit ? RESULT_CODE_RAY_ABSORBED : result;
}

#else

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
    ResultCode result = propagateSampled(
        ray, dist,
        hit,
        params.propagation,
        idx, dim);
    if (hit) return RESULT_CODE_RAY_ABSORBED;
    if (result < 0) return result;

    //trace shadow ray for NEE
    traceShadowRay(ray, cam, idx, dim);
    return RESULT_CODE_SUCCESS;
}

#endif

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.batchSize) return;

    //sample wavelength
    float lambdaContrib;
    float lambda = sampleWavelength(lambdaContrib, idx, dim);

    //optionally, sample direct light contribution
    #ifndef DISABLE_DIRECT_LIGHTING
    sampleDirect(params.propagation.maxTime, idx, dim);
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
