#ifndef _INCLUDE_RAY_COMBINE
#define _INCLUDE_RAY_COMBINE

#include "ray.glsl"
#include "ray.scatter.glsl"
#include "ray.util.glsl"
#include "result.glsl"
#include "response.common.glsl"

#ifdef POLARIZATION

/**
 * Combines the given rays creating a hit. Returns RESULT_CODE_SUCCES, if the
 * created hit is within bounds.
 * Assumes the rays are already anti-parallel, i.e. it will not perform any
 * scattering to align them.
*/
ResultCode combineRaysAligned(
    inout PolarizedBackwardRay ray, ///< Ray from camera
    const SourceRay source,         ///< Sampled ray from light source
    const CameraHit hit,            ///< Sampled camera hit
    const PropagateParams params,   ///< Propagation parameters
    out HitItem result              ///< Created hit
) {
    //apply attenuation (skips boundary checks from propagateRay())
    float dist = distance(ray.state.position, source.position);
    ResultCode code = updateRay(ray, dist, params);
    if (code < 0)
        return code; //hit outside bounds
    
    //align polarization reference frames
    mat4 align = alignPolRef(source.direction, source.polRef, ray.polRef);

    //combine rays
    float contrib = source.contrib * getContrib(ray);
    vec4 stokes = ray.mueller * align * source.stokes;
    float time = source.startTime + ray.state.time;
    //normalize stokes
    contrib *= stokes.x;
    stokes /= stokes.x;

    //create HitItem
    result = HitItem(
        hit.position,
        hit.direction,
        hit.normal,
        stokes,
        hit.polRef,
        ray.state.wavelength,
        time,
        contrib
    );
    //successful if we have any contribution
    return contrib > 0.0 ? RESULT_CODE_SUCCESS : RESULT_CODE_RAY_ABSORBED;
}
ResultCode combineRaysAligned(
    inout PolarizedForwardRay ray,  ///< Ray from light source
    const CameraRay camera,         ///< Sample camera ray
    const PropagateParams params,   ///< Propagation params
    out HitItem result              ///< Created hit
) {
    //apply attenuation (skips boundary checks from propagateRay())
    float dist = distance(ray.state.position, camera.position);
    ResultCode code = updateRay(ray, dist, params);
    if (code < 0)
        return code; //hit outside bounds
    
    //align polarization reference frames
    mat4 align = alignPolRef(ray.state.direction, ray.polRef, camera.polRef);

    //combine rays
    float contrib = camera.contrib * getContrib(ray);
    vec4 stokes = camera.mueller * align * ray.stokes;
    float time = ray.state.time + camera.timeDelta;
    //normalize stokes
    contrib *= stokes.x;
    stokes /= stokes.x;

    //create HitItem
    result = HitItem(
        camera.hit.position,
        camera.hit.direction,
        camera.hit.normal,
        stokes,
        camera.hit.polRef,
        ray.state.wavelength,
        time,
        contrib
    );
    //successful if we have any contribution
    return contrib > 0.0 ? RESULT_CODE_SUCCESS : RESULT_CODE_RAY_ABSORBED;
}

#else

ResultCode combineRaysAligned(
    inout UnpolarizedBackwardRay ray,
    const SourceRay source,
    const CameraHit hit,
    const PropagateParams params,
    out HitItem result
) {
    //apply attenuation (skips boundary checks from propagateRay())
    float dist = distance(ray.state.position, source.position);
    ResultCode code = updateRay(ray, dist, params);
    if (code < 0)
        return code; //hit outside bounds
    
    //combine rays
    float contrib = source.contrib * getContrib(ray);
    float time = source.startTime + ray.state.time;

    //create HitItem
    result = HitItem(
        hit.position,
        hit.direction,
        hit.normal,
        ray.state.wavelength,
        time,
        contrib
    );
    //successful if we have any contribution
    return contrib > 0.0 ? RESULT_CODE_SUCCESS : RESULT_CODE_RAY_ABSORBED;
}
ResultCode combineRaysAligned(
    inout UnpolarizedForwardRay ray,
    const CameraRay camera,
    const PropagateParams params,
    out HitItem result
) {
    //apply attenuation (skips boundary checks from propagateRay())
    float dist = distance(ray.state.position, camera.position);
    ResultCode code = updateRay(ray, dist, params);
    if (code < 0)
        return code; //hit outside bounds
    
    //combine rays
    float contrib = camera.contrib * getContrib(ray);
    float time = ray.state.time + camera.timeDelta;

    //create HitItem
    result = HitItem(
        camera.hit.position,
        camera.hit.direction,
        camera.hit.normal,
        ray.state.wavelength,
        time,
        contrib
    );
    //successful if we have any contribution
    return contrib > 0.0 ? RESULT_CODE_SUCCESS : RESULT_CODE_RAY_ABSORBED;
}

#endif //#ifdef POLARIZATION

/**
 * Combines the given rays creating a hit. Returns RESULT_CODE_SUCCES, if the
 * created hit is within bounds.
 * Performs ONE extra scatter to align the ray with the light source, but
 * assumes the given SourceRay points to the ray's current position.
*/
ResultCode combineRays(
    BackwardRay ray,                ///< Ray from camera
    const SourceRay source,         ///< Sampled ray from light source
    const CameraHit hit,            ///< Sampled camera hit
    const PropagateParams params,   ///< Propagation parameters
    out HitItem result              ///< Created hit
) {
    //scatter ray to point towards source
    scatterRay(ray, -source.direction);

    //combine
    return combineRaysAligned(ray, source, hit, params, result);
}
ResultCode combineRays(
    ForwardRay ray,
    const CameraRay camera,
    const PropagateParams params,
    out HitItem result
) {
    //scatter ray to point towards camera
    scatterRay(ray, -camera.direction);

    //combine
    return combineRaysAligned(ray, camera, params, result);
}

#endif
