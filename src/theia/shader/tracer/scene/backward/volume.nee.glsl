#ifndef _INCLUDE_VOLUME_NEE_MIS_IMPL
#define _INCLUDE_VOLUME_NEE_MIS_IMPL

//user provided code
#include "response.glsl"
#include "source.glsl"
#include "volume.glsl"

#include "tracer/propagate/backward.glsl"
#include "tracer/propagate/combine.glsl"

bool isVisible(vec3 a, vec3 b) {
    float dist = distance(a, b);
    if (dist <= 0.0) return true; //nothing in between to block

    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery,
        accelerationStructureEXT(params.tlas),
        gl_RayFlagsTerminateOnFirstHitEXT,      //flags
        0xFF,                                   //cull mask
        a,                                      //origin
        0.0,                                    //t_min
        normalize(b - a),                       //direction
        dist                                    //t_max
    );
    rayQueryProceedEXT(rayQuery);
    return rayQueryGetIntersectionTypeEXT(rayQuery, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT;
}

void traceShadowRay(
    BackwardRay ray,
    float tStep,
    const CameraHit cam,
    float pVolume,
    #ifdef LightNeeContext
    LightNeeContext lightNeeContext,
    #endif
    inout uint dim
) {
    //propagate to scatter point
    //no visibility check needed as it's ensured by limiting the sampling earlier
    ResultCode result = propagate(
        ray, tStep, false, false, params.propagation, gl_LaunchIDEXT.x, dim);
    if (result < 0) return;

    //sample light source
    //
    //we use the balance heuristic with MIS. This turns the MIS weight normalization
    //into a plain sum, making it easier to calculate them across API boundaries
    //this is necessary to allow lights to cancel factors they want to get rid
    //of (like 1/r^2 in point light sources) in their contribution while still
    //applying the missing factor to the volume pdf to get a correct estimate.
    ForwardRay source = sampleLightMIS(
        ray.position, ray.direction, tStep,
        ray.wavelength, ray.mediumIdx,
        pVolume,
        #ifdef LightNeeContext
        lightNeeContext,
        #endif
        gl_LaunchIDEXT.x, dim
    );

    //cast shadow ray
    if (!isVisible(ray.position, source.position)) return;

    //create hit by combining light and camera ray
    HitItem hit;
    result = combineRays(
        ray, source, cam,
        params.propagation,
        hit, gl_LaunchIDEXT.x, dim);
    if (result >= 0) {
        response(hit, gl_LaunchIDEXT.x, dim);
        onEvent(ray, RESULT_CODE_RAY_DETECTED, gl_LaunchIDEXT.x, dim);
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
    const CameraHit camHit,
    float tMax,
    inout uint dim
) {
    //to be consistent with the proxy variant, deterministically advance dim later
    uint origDim = dim;

    //query light source for a step size
    #ifdef LightNeeContext
    LightNeeContext lightNeeContext;
    #endif
    float tLight = sampleLightNeeStepSize(
        ray.position,
        ray.direction,
        tMax,
        ray.wavelength,
        ray.mediumIdx,
        #ifdef LightNeeContext
        lightNeeContext,
        #endif
        gl_LaunchIDEXT.x, dim
    );
    //evaluate volume model pdf needed for MIS weight
    float pVolume = evalVolumeNeeStepSize(ray, tMax, tLight);
    //trace shadow ray
    traceShadowRay(
        ray,
        tLight,
        camHit,
        pVolume,
        #ifdef LightNeeContext
        lightNeeContext,
        #endif
        dim
    );

    //query volume model for a step size
    float tVolume = sampleVolumeNeeStepSize(
        ray,
        tMax,
        pVolume,
        gl_LaunchIDEXT.x, dim
    );
    //evaluation of light pdf is handled by light source itself
    //-> can directly trace shadow ray
    traceShadowRay(
        ray,
        tVolume,
        camHit,
        pVolume,
        #ifdef LightNeeContext
        lightNeeContext,
        #endif
        dim
    );

    //advance RNG deterministically
    dim = origDim + VOLUME_IN_SCATTERING_MIS_DIM_STRIDE;
}

#endif
