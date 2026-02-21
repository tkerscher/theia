#include "result.glsl"
#include "material.glsl"
#include "scene/types.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "surface.glsl"
#include "response.glsl"

#include "tracer/scene/volume/proxy.glsl"
#include "tracer/propagate/forward.glsl"
#include "scene/intersect.glsl"

#include "tracer/scene/forward/io.glsl"

//mapping from TLAS instance -> objectId
readonly buffer ObjectIdMap{ int objectIdMap[]; };

layout(location = 1) rayPayloadInEXT NeeData neeData;
hitAttributeEXT vec2 barys; //filled by default intersection shader

void main() {
    uint dim = neeData.dim;
    //resolve hit
    SurfaceHit hit;
    ResultCode result = resolveIntersection(neeData.ray.mediumIdx, barys, hit);
    if (result < 0) return;

    //check if we hit a detector
    bool isDet = (hit.flags & MATERIAL_DETECTOR_BIT) != 0;
    if (!isDet) return;
    //check if we have a filter on acceptable targets
    //0x80000000 marks no filter (int32 min value)
    int objectId = objectIdMap[gl_InstanceID];
    if (params.targetId != 0x80000000 && params.targetId != objectId) return;

    //propagate ray to hit
    result = propagateToHit(
        neeData.ray,
        hit.worldPos,
        hit.rayNrm,
        false,
        params.propagation,
        gl_LaunchIDEXT.x, dim
    );
    if (result < 0) return;

    //if the surface model requests it, prepare interaction
    #ifdef SurfaceProperties
    SurfaceProperties props = prepareSurface(
        neeData.ray, hit, gl_LaunchIDEXT.x, dim
    );
    #endif

    //process surface hit
    HitItem item;
    bool success = processSurfaceTargetHit(
        neeData.ray,
        hit,
        #ifdef SurfaceProperties
        props,
        #endif
        objectId,
        item,
        gl_LaunchIDEXT.x, dim
    );
    item.contrib *= neeData.weight;
    //process hit item if successful
    if (success && item.contrib > 0.0) {
        response(item, gl_LaunchIDEXT.x, dim);
    }
}
