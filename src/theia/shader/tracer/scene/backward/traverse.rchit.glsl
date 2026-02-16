#include "result.glsl"
#include "material.glsl"
#include "scene/types.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "surface.glsl"

#include "tracer/scene/volume/proxy.backward.glsl"
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
    traceData.result = resolveIntersection(
        traceData.ray.mediumIdx,
        attribs,
        hit
    );
    if (traceData.result < 0) return;

    //propagate ray to hit
    traceData.result = propagateSampledToHit(
        traceData.ray,
        hit.worldPos,
        hit.rayNrm,
        params.propagation,
        gl_LaunchIDEXT.x,
        traceData.dim
    );
    if (traceData.result < 0) return;

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
