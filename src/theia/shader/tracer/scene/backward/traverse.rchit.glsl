#include "result.glsl"
#include "material.glsl"
#include "scene/types.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "surface.glsl"

#include "tracer/scene/volume/proxy.glsl"
#include "tracer/propagate/backward.glsl"
#include "scene/intersect.glsl"

uniform TraceParams {
    uvec2 tlas;
    PropagationParams propagation;
    uint batchSize;
} params;

struct TraceData{
    BackwardRay ray;
    uint dim;
    ResultCode result;
};
layout(location = 0) rayPayloadInEXT TraceData traceData;
hitAttributeEXT vec2 attribs; //default implementation reports barys here

#ifndef SURFACE_MODEL_SPECULAR

#include "tracer/scene/backward/nee.glsl"

void traceNee(
    BackwardRay ray,
    const SurfaceHit hit,
    #ifdef SurfaceProperties
    const SurfaceProperties props,
    #endif
    inout uint dim
) {
    //sample light source
    ForwardRay source = sampleLight(
        ray.position,
        hit.rayNrm,
        ray.wavelength,
        ray.mediumIdx,
        gl_LaunchIDEXT.x, dim
    );

    //scatter ray towards source
    ResultCode result = surfaceScatterRay(
        ray, hit,
        #ifdef SurfaceProperties
        props,
        #endif
        -source.direction,
        gl_LaunchIDEXT.x, dim
    );
    if (result < 0) return;
    //extra cosine from geometric term (projected area in radiance defintion)
    ray.lin_contrib *= abs(dot(-source.direction, hit.rayNrm));

    //trace NEE
    traceNee(ray, source, params.propagation, params.tlas, dim);
}

#endif

void main() {
    //resolve hit
    SurfaceHit hit;
    ResultCode intersectResult = resolveIntersection(
        traceData.ray.mediumIdx,
        attribs,
        hit
    );
    //propagate even if intersection failed to get better data for debugging
    //intersect should only fail with ERROR_CODE_MEDIA_MISMATCH so the actual
    //intersection should still be good
    traceData.result = propagateSampledToHit(
        traceData.ray,
        hit.worldPos,
        hit.rayNrm,
        params.propagation,
        gl_LaunchIDEXT.x,
        traceData.dim
    );
    if (intersectResult < 0)
        traceData.result = intersectResult;
    if (traceData.result < 0) return;

    //skip any surface sampling if surface is marked as absorber and return early
    if ((hit.flags & MATERIAL_BLACK_BODY_BIT) != 0) {
        traceData.result = RESULT_CODE_RAY_ABSORBED;
        return;
    }

    //if the surface model requests it, prepare interaction
    #ifdef SurfaceProperties
    SurfaceProperties props = prepareSurface(
        traceData.ray, hit, gl_LaunchIDEXT.x, traceData.dim
    );
    #endif

    //on non-specular surfaces, we can perform NEE
    #ifndef SURFACE_MODEL_SPECULAR
    traceNee(
        traceData.ray, hit,
        #ifdef SurfaceProperties
        props,
        #endif
        traceData.dim
    );
    #endif

    //sample surface interaction
    traceData.result = sampleSurfaceInteraction(
        traceData.ray,
        hit,
        #ifdef SurfaceProperties
        props,
        #endif
        gl_LaunchIDEXT.x,
        traceData.dim
    );
}
