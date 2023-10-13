#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_ray_query : require

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require

#include "wavefront.items.glsl"

layout(local_size_x = 32) in;

layout(scalar) readonly buffer RayQueue{
    uint rayCount;
    RayItem rayItems[];
};
layout(scalar) writeonly buffer IntersectionQueue{
    uint intersectionCount;
    IntersectionItem intersectionItems[];
};
layout(scalar) writeonly buffer VolumeScatterQueue{
    uint volumeCount;
    VolumeScatterItem volumeItems[];
};

layout(scalar) readonly buffer RNGBuffer{ float u[]; };
uniform accelerationStructureEXT tlas;

layout(scalar) uniform TraceParams{
    // for transient rendering, we won't importance sample the media
    float scatterCoefficient;

    float maxTime;
    vec3 lowerBBoxCorner;
    vec3 upperBBoxCorner;
} params;

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= rayCount)
        return;
    Ray ray = rayItems[idx].ray;

    //just to be safe
    vec3 dir = normalize(ray.direction);
    //sample distance
    float dist = -log(1.0 - u[ray.rngIdx++]) / params.scatterCoefficient;

    //trace ray
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF,   //mask -> hit anything
        ray.position,
        0.0,    //t_min; self-intersections are handled in surface scatter kernel
        dir,
        dist/*max(1e-6, dist)*/);
    rayQueryProceedEXT(rayQuery);

    //check if we hit anything
    bool hit = rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT;

    if (hit) {
        //we hit something -> create an intersection item

        //query all necessary information
        int instanceId   = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
        int customId     = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
        int triangleId   = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
        vec2 baryc       = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
        mat4x3 obj2World = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
        mat4x3 world2Obj = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);

        //count how many items we will be adding in this subgroup
        uint n = subgroupAdd(1);
        //elect one invocation to update queue counter
        uint oldCount = 0;
        if (subgroupElect()) {
            oldCount = atomicAdd(intersectionCount, n);
        }
        //only the first(i.e. elected) one has correct oldCount value -> broadcast
        oldCount = subgroupBroadcastFirst(oldCount);
        //no we effectevily reserved us the range [oldCount..oldcount+n-1]

        //order the active invocations so each can write at their own spot
        uint id = subgroupExclusiveAdd(1);
        //create and save item
        intersectionItems[oldCount + id] = IntersectionItem(
            ray,
            rayItems[idx].targetIdx,

            instanceId,
            customId,
            triangleId,
            baryc,
            obj2World,
            world2Obj
        );
    }
    else {
        //we hit nothing -> create a volume scatter item

        //TODO: do boundary check here?

        //count how many items we will be adding in this subgroup
        uint n = subgroupAdd(1);
        //elect one invocation to update counter in queue
        uint oldCount = 0;
        if (subgroupElect()) {
            oldCount = atomicAdd(volumeCount, n);
        }
        //only the first(i.e. elected) one has correct oldCount value -> broadcast
        oldCount = subgroupBroadcastFirst(oldCount);
        //no we effectevily reserved us the range [oldCount..oldcount+n-1]

        //order the active invocations so each can write at their own spot
        uint id = subgroupExclusiveAdd(1);
        //create and save item
        volumeItems[oldCount + id] = VolumeScatterItem(
            ray,
            rayItems[idx].targetIdx,
            
            dist
        );
    }
}
