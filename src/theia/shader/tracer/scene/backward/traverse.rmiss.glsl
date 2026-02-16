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

struct NeeData {
    BackwardRay ray;
    uint dim;
};
layout(location = 1) rayPayloadEXT NeeData neeData;

#ifndef VOLUME_MODEL_NO_SCATTERING

void neeEstimate(
    BackwardRay ray,
    inout uint dim
) {
    //the miss shader will later resample the light source with the current rng
    //state so we do not have to put the sampled forward ray in the payload
    neeData.ray = ray;
    neeData.dim = dim;
    //sample light source
    ForwardRay source = sampleLight(
        ray.position, vec3(0.0),
        ray.wavelength,
        ray.mediumIdx,
        gl_LaunchIDEXT.x, dim
    );

    //is there even a chance for successfull connection?
    if (ray.mediumIdx != ray.mediumIdx) return;

    //trace shadow ray
    traceRayEXT(
        accelerationStructureEXT(params.tlas),
        gl_RayFlagsTerminateOnFirstHitEXT,
        0xFF,                                           //cull mask
        1, /* same as direct; both want None */         //sbt offset
        0,                                              //sbt stride
        MISS_STRIDE * getVolumeIdx(ray.mediumIdx) + 1,  //miss index
        ray.position,                                   //origin
        0.0,                                            //t_min
        -source.direction,                              //direction
        distance(source.position, ray.position),        //t_max
        1                                               //payload location
    );
    //we can safely ignore rng dim. We will advance it in the main tracing loop
    //(saves a bit of memory bandwidth)
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
