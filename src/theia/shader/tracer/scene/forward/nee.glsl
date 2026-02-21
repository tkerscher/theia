#ifndef _INCLUDE_TRACER_SCENE_FORWARD_NEE
#define _INCLUDE_TRACER_SCENE_FORWARD_NEE

#include "target_guide/common.glsl"
#include "target.glsl"

layout(location = 1) rayPayloadEXT NeeData neeData;

void traceNEE(ForwardRay ray, float dist, float weight, uvec2 tlas, inout uint dim) {
    //don't bother tracing if we don't expect any contribution
    if (weight == 0) return;

    //prepare payload
    neeData.ray = ray;
    neeData.weight = weight;
    neeData.dim = dim;

    //trace ray
    traceRayEXT(
        accelerationStructureEXT(tlas),
        gl_RayFlagsOpaqueEXT,
        0xFF,                               //cull mask
        1,                                  //sbt offset
        0,                                  //sbt stride
        0,                                  //miss index
        ray.position,                       //origin
        0.0,                                //t_min
        ray.direction,                      //direction
        dist,                               //t_max
        1                                   //payload location
    );

    //read back result (TODO: make this deterministic)
    dim = neeData.dim;
}

void traceNEE(ForwardRay ray, uvec2 tlas, inout uint dim) {
    //check if we have a chance to hit target
    TargetGuideSample guideSample = evalTargetGuide(ray.position, ray.direction);
    if (guideSample.prob == 0) return;
    traceNEE(ray, guideSample.dist, 1.0, tlas, dim);
}

#endif
