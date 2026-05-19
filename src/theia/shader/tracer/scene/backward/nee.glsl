#ifndef _INCLUDE_SCENE_BACKWARD_TRACER_NEE
#define _INCLUDE_SCENE_BACKWARD_TRACER_NEE

#include "callback.glsl"
#include "camera.glsl"
#include "photon.glsl"
#include "response.glsl"
#include "source.glsl"

void traceNee(
    BackwardRay ray,
    ForwardRay source,
    const PropagationParams params,
    uvec2 tlas,
    inout uint dim
) {
    //is there even a chance for successfull connection?
    if (ray.mediumIdx != ray.mediumIdx) return;

    float dist = distance(ray.position, source.position);
    //trace shadow ray
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery,
        accelerationStructureEXT(tlas),
        gl_RayFlagsTerminateOnFirstHitEXT,  //flags
        0xFF,                               //cull mask
        ray.position,                       //origin
        0.0,                                //t_min
        ray.direction,                      //direction
        dist                                //t_max
    );
    rayQueryProceedEXT(rayQuery);
    if (
        rayQueryGetIntersectionTypeEXT(rayQuery, true) !=
        gl_RayQueryCommittedIntersectionNoneEXT
    ) return; //shadowed
    
    //propagate ray towards source
    ResultCode result = propagate(
        ray,
        dist,
        true, //treat as if we have hit the light source for correct propagation
        false,
        params,
        gl_LaunchIDEXT.x, dim
    );
    if (result < 0) return;

    //resample camera hit (so we do not need to cache it in memory)
    uint reDim = CAMERA_SAMPLE_RNG_DIM;
    float lambdaContrib;
    float lambda = sampleWavelength(lambdaContrib, gl_LaunchIDEXT.x, reDim);
    CameraHit camHit;
    sampleCameraRay(lambda, camHit, gl_LaunchIDEXT.x, reDim);

    //combine
    HitItem hit;
    result = combineRaysAligned(source, ray, camHit, hit);
    if (result < 0) return;
    //check time limit if applicable
    #ifdef RAY_TRANSIENT
    if (hit.time > params.maxTime) return;
    #endif

    //process hit
    if (hit.contrib > 0.0) {
        response(hit, gl_LaunchIDEXT.x, dim);
        onEvent(ray, RESULT_CODE_RAY_DETECTED, gl_LaunchIDEXT.x, dim);
    }
}

#endif
