#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_ray_query : require

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require

#include "rng.glsl"
#include "wavefront.common.glsl"

layout(local_size_x = LOCAL_SIZE) in;

layout(scalar) readonly buffer RayQueueBuffer {
    uint rayCount;
    RayQueue rayQueue;
};
layout(scalar) writeonly buffer IntersectionQueueBuffer {
    uint intersectionCount;
    IntersectionQueue intersectionQueue;
};
layout(scalar) writeonly buffer VolumeScatterQueueBuffer {
    uint volumeCount;
    VolumeScatterQueue volumeQueue;
};

uniform accelerationStructureEXT tlas;

layout(scalar) uniform Params {
    TraceParams params;
};

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= rayCount)
        return;
    //load ray
    LOAD_RAY(ray, rayQueue, idx)

    //just to be safe
    vec3 dir = normalize(ray.direction);
    //sample distance
    float u = random(ray.rngStream, ray.rngCount++);
    float dist = -log(1.0 - u) / params.scatterCoefficient;

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
        idx = oldCount + id;
        SAVE_RAY(ray, intersectionQueue.rays, idx)

        intersectionQueue.geometryIdx[idx] = instanceId;
        intersectionQueue.customIdx[idx] = customId;
        intersectionQueue.triangleIdx[idx] = triangleId;
        intersectionQueue.baryU[idx] = baryc.x;
        intersectionQueue.baryV[idx] = baryc.y;
        //mat4x3 obj2World matrix ij -> i-th column, j-th row
        intersectionQueue.obj2World00[idx] = obj2World[0][0];
        intersectionQueue.obj2World01[idx] = obj2World[0][1];
        intersectionQueue.obj2World02[idx] = obj2World[0][2];
        intersectionQueue.obj2World10[idx] = obj2World[1][0];
        intersectionQueue.obj2World11[idx] = obj2World[1][1];
        intersectionQueue.obj2World12[idx] = obj2World[1][2];
        intersectionQueue.obj2World20[idx] = obj2World[2][0];
        intersectionQueue.obj2World21[idx] = obj2World[2][1];
        intersectionQueue.obj2World22[idx] = obj2World[2][2];
        intersectionQueue.obj2World30[idx] = obj2World[3][0];
        intersectionQueue.obj2World31[idx] = obj2World[3][1];
        intersectionQueue.obj2World32[idx] = obj2World[3][2];
        //mat4x3 world2Obj matrix ij -> i-th column, j-th row
        intersectionQueue.world2Obj00[idx] = world2Obj[0][0];
        intersectionQueue.world2Obj01[idx] = world2Obj[0][1];
        intersectionQueue.world2Obj02[idx] = world2Obj[0][2];
        intersectionQueue.world2Obj10[idx] = world2Obj[1][0];
        intersectionQueue.world2Obj11[idx] = world2Obj[1][1];
        intersectionQueue.world2Obj12[idx] = world2Obj[1][2];
        intersectionQueue.world2Obj20[idx] = world2Obj[2][0];
        intersectionQueue.world2Obj21[idx] = world2Obj[2][1];
        intersectionQueue.world2Obj22[idx] = world2Obj[2][2];
        intersectionQueue.world2Obj30[idx] = world2Obj[3][0];
        intersectionQueue.world2Obj31[idx] = world2Obj[3][1];
        intersectionQueue.world2Obj32[idx] = world2Obj[3][2];
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
        idx = oldCount + id;
        SAVE_RAY(ray, volumeQueue.rays, idx)
        volumeQueue.dist[idx] = dist;
    }
}
