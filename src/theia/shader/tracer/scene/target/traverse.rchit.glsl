#include "result.glsl"
#include "material.glsl"
#include "scene/types.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "surface.glsl"
#include "response.glsl"

#include "tracer/scene/target/config.glsl"
#include "tracer/scene/volume/proxy.glsl"
#include "tracer/scene/target/propagate.glsl"
#include "scene/intersect.glsl"

#include "tracer/scene/target/io.glsl"

#ifndef DISABLE_NEE
#include "tracer/scene/target/nee.glsl"
#endif

//mapping from TLAS instance -> objectId
readonly buffer ObjectIdMap{ int objectIdMap[]; };

layout(location = 0) rayPayloadInEXT TraceData traceData;
hitAttributeEXT vec2 barys; //filled by default intersection shader

void main() {
    //we do not need to report rng state back as we advance it in the trace loop
    uint dim = traceData.dim;
    //resolve
    SurfaceHit hit;
    traceData.result = resolveIntersection(traceData.ray.mediumIdx, barys, hit);
    if (traceData.result < 0) return;

    //propagate ray to hit
    traceData.result = propagateSampledToHit(
        traceData.ray,
        hit.worldPos,
        hit.rayNrm,
        params.propagation,
        gl_LaunchIDEXT.x, dim
    );
    if (traceData.result < 0) return;

    //if the surface model requests it, prepare interaction
    #ifdef SurfaceProperties
    SurfaceProperties props = prepareSurface(
        traceData.ray, hit, gl_LaunchIDEXT.x, dim
    );
    #endif

    //create response if we hit target and the tracer allows it
    //if we do NEE, we will create responses exclusively through it and not in
    //the tracing loop. Otherwise, do it here.
    #ifdef DISABLE_NEE
    //did we hit a target?
    bool isTarget = (hit.flags & MATERIAL_TARGET_BIT) != 0;
    //do we have a filter on the objectId? (0x80000000 marks no filter)
    int objectId = objectIdMap[gl_InstanceID];
    bool filtered = params.targetId != 0x80000000 && params.targetId != objectId;
    if (isTarget && !filtered) {
        HitItem item;
        bool success = processSurfaceTargetHit(
            traceData.ray,
            hit,
            #ifdef SurfaceProperties
            props,
            #endif
            objectId,
            item,
            gl_LaunchIDEXT.x, dim
        );
        if (success) {
            response(item, gl_LaunchIDEXT.x, dim);
        }
    }
    #endif

    //skip surface sampling if it was marked as absorber and return early
    //note that absorber may still create a response, so do not skip that step
    if ((hit.flags & MATERIAL_BLACK_BODY_BIT) != 0) {
        traceData.result = RESULT_CODE_RAY_ABSORBED;
        return;
    }

    //sample surface interaction
    traceData.result = sampleSurfaceInteraction(
        traceData.ray,
        hit,
        #ifdef SurfaceProperties
        props,
        #endif
        gl_LaunchIDEXT.x, dim
    );
    if (traceData.result < 0) return;

    //if NEE enabled, check if we can extend ray to target guide
    #ifndef DISABLE_NEE
    traceNEE(traceData.ray, params.tlas, dim);
    #endif

    //TODO: MIS for non specular surfaces
}
