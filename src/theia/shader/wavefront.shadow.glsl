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

#include "scatter.surface.glsl"
#include "wavefront.items.glsl"

layout(local_size_x = 32) in;

layout(scalar) readonly buffer ShadowQueue{
    uint shadowCount;
    ShadowRayItem shadowItems[];
};
layout(scalar) writeonly buffer ResponseQueue{
    uint responseCount;
    ResponseItem responseItems[];
};

layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer Vertex {
    vec3 position;
    vec3 normal;
};
layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer Index {
    ivec3 idx;
};
struct Geometry{
    Vertex vertices;    // &vertices[0]
    Index indices;      // &indices[0]
    Material material;
};
layout(scalar) buffer Geometries{ Geometry geometries[]; };
uniform accelerationStructureEXT tlas;

layout(scalar) uniform TraceParams{
    // for transient rendering, we won't importance sample the media
    float scatterCoefficient;

    float maxTime;
    vec3 lowerBBoxCorner;
    vec3 upperBBoxCorner;
} params;

void main() {
    //Range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= shadowCount)
        return;
    ShadowRayItem item = shadowItems[idx];
    Ray ray = item.ray;
    
    //check if we can hit the detector
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF,
        ray.position,
        0.0,
        ray.direction,
        item.dist);
    rayQueryProceedEXT(rayQuery);

    //Did we hit anything? (we might have missed the detector)
    if (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionTriangleEXT)
        return;
    //check if we hit the target and not something else
    int customIdx = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    int geometryIdx = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
    Geometry geom = geometries[geometryIdx];
    if (customIdx != item.targetIdx && (geom.material.flags & MATERIAL_TARGET_BIT) != 0)
        return;
    
    //We indeed hit the detector -> collect all infromations about the hit

    //fetch hit properties
    int triangleIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    vec2 barys = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    mat4x3 obj2World = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
    mat4x3 world2Obj = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
    //fetch hit triangle
    ivec3 index = geom.indices[triangleIdx].idx;
    Vertex v0 = geom.vertices[index.x];
    Vertex v1 = geom.vertices[index.y];
    Vertex v2 = geom.vertices[index.z];
    //reconstruct hit position
    precise vec3 e1 = v1.position - v0.position;
    precise vec3 e2 = v2.position - v0.position;
    precise vec3 objPos = v0.position + fma(vec3(barys.x), e1, barys.y * e2);
    //calc exact distance
    mat4x3 m = obj2World;
    precise vec3 worldPos;
    worldPos.x = m[3][0] + fma(m[0][0], objPos.x, fma(m[1][0], objPos.y, m[2][0] * objPos.z));
    worldPos.y = m[3][1] + fma(m[0][1], objPos.x, fma(m[1][1], objPos.y, m[2][1] * objPos.z));
    worldPos.z = m[3][2] + fma(m[0][2], objPos.x, fma(m[1][2], objPos.y, m[2][2] * objPos.z));
    float dist = length(worldPos - ray.position);
    //interpolate normal
    precise vec3 n1 = v1.normal - v0.normal;
    precise vec3 n2 = v2.normal - v0.normal;
    precise vec3 objNormal = v0.normal + fma(vec3(barys.x), n1, barys.y * n2);
    objNormal = normalize(objNormal);
    //transform ray direction to object space
    vec3 objDir = normalize(mat3(world2Obj) * ray.direction);

    //create hits
    bool anyBelowMaxTime = false;
    float lambda = params.scatterCoefficient;
    PhotonHit hits[N_PHOTONS];
    for (int i = 0; i < N_PHOTONS; ++i) {
        //calculate reflectance
        float r = reflectance(geom.material, ray.photons[i], objNormal, objDir);
        //we absorb -> attenuate by transmission
        ray.photons[i].T_lin *= (1.0 - r);

        //since we hit something the prob for the distance is to sample at
        //least dist -> p(d>=dist) = exp(-lambda*dist)
        //attenuation is simply Beer's law
        float mu_e = ray.photons[i].constants.mu_e;
        //write out multplication in hope to prevent catastrophic cancelation
        ray.photons[i].T_log += lambda * dist - mu_e * dist;

        //update travel time
        ray.photons[i].travelTime += dist / ray.photons[i].constants.vg;
        //bound check
        if (ray.photons[i].travelTime <= params.maxTime)
            anyBelowMaxTime = true;
        
        //create hit
        hits[i] = createHit(ray.photons[i]);
    }
    //bounds check: max time
    if (!anyBelowMaxTime)
        return;
    
    //create response item

    //count how many items we will be adding in this subgroup
    uint n = subgroupAdd(1);
    //elect one invocation to update counter in queue
    uint oldCount = 0;
    if (subgroupElect()) {
        oldCount = atomicAdd(responseCount, n);
    }
    //only the first(i.e. elected) one has correct oldCount value -> broadcast
    oldCount = subgroupBroadcastFirst(oldCount);
    //no we effectevily reserved us the range [oldCount..oldcount+n-1]

    //order the active invocations so each can write at their own spot
    uint id = subgroupExclusiveAdd(1);
    //create response item
    responseItems[oldCount + id] = ResponseItem(
        objPos, objDir, objNormal, item.targetIdx, hits
    );
}
