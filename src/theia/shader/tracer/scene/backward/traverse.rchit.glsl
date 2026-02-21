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
    
    //TODO: Allow non-specular surfaces. In that case, we can sample the light for NEE
}
