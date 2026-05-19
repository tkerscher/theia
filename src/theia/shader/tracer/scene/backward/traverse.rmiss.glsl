#include "result.glsl"
#include "material.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "source.glsl"
#include "volume.glsl"

#include "tracer/scene/volume/index.glsl"
#include "tracer/propagate/backward.glsl"

uniform TraceParams {
    uvec2 tlas;
    PropagationParams propagation;
} params;

struct TraceData{
    BackwardRay ray;
    uint dim;
    ResultCode result;
};
layout(location = 0) rayPayloadInEXT TraceData traceData;

#ifndef VOLUME_MODEL_NO_SCATTERING

#include "tracer/scene/backward/nee.glsl"

void neeEstimate(
    BackwardRay ray,
    inout uint dim
) {
    //sample light source
    ForwardRay source = sampleLight(
        ray.position, vec3(0.0),
        ray.wavelength,
        ray.mediumIdx,
        gl_LaunchIDEXT.x, dim
    );
    
    //scatter ray towards source
    ResultCode result = volumeScatterRay(
        ray, -source.direction,
        gl_LaunchIDEXT.x, dim  
    );
    if (result < 0) return;

    //trace NEE ray
    traceNee(ray, source, params.propagation, params.tlas, dim);
}

#endif

void main() {
    //propagate ray
    traceData.result = propagateSampled(
        traceData.ray,
        gl_RayTmaxEXT,
        false,
        params.propagation,
        gl_LaunchIDEXT.x, traceData.dim
    );
    if (traceData.result < 0) return;

    //do not bother with NEE if the volume does not support it anyway
    #ifndef VOLUME_MODEL_NO_SCATTERING
    //NEE estimate to complete path
    neeEstimate(traceData.ray, traceData.dim);
    #endif

    //sample new direction
    traceData.result = sampleVolumeInteraction(
        traceData.ray,
        gl_LaunchIDEXT.x,
        traceData.dim
    );
}
