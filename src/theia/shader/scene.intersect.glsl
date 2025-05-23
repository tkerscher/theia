#ifndef _INCLUDE_SCENE_INTERSECT
#define _INCLUDE_SCENE_INTERSECT

#include "math.glsl"
#include "ray.glsl"
#include "result.glsl"
#include "scene.types.glsl"

layout(scalar) buffer Geometries{ Geometry geometries[]; };
//Top level acceleration structure containing the scene
uniform accelerationStructureEXT tlas;

/*
 * process a given ray query and calculates the hit's position and surface
 * normal in both world and object space. Returns true, if successfull, false
 * otherwise.
 *
 * Note: Normals always points outwards.
 *
 * Note: This version does not interpolate normals, but use it to determine the
 *       orientation of the material (inward/outward).
*/
ResultCode processRayQuery(
    const RayState ray,     ///< Current propagation state
    rayQueryEXT rayQuery,   ///< Query used to trace scene
    out SurfaceHit hit      ///< Structure describing the hit
) {
    //check if we hit anything
    hit.valid = (
        rayQueryGetIntersectionTypeEXT(rayQuery, true)
        == gl_RayQueryCommittedIntersectionTriangleEXT
    );
    if (!hit.valid)
        return RESULT_CODE_RAY_MISSED;
      
    //fetch info about intersection
    int instanceId = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
    int triangleId = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    vec2 barys = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    hit.customId = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    //reconstruct hit triangle
    Geometry geom = geometries[instanceId];
    ivec3 index = geom.indices[triangleId].idx;
    Vertex v0 = geom.vertices[index.x];
    Vertex v1 = geom.vertices[index.y];
    Vertex v2 = geom.vertices[index.z];
    precise vec3 e1 = v1.position - v0.position;
    precise vec3 e2 = v2.position - v0.position;
    hit.objPos = v0.position + fma(vec3(barys.x), e1, barys.y * e2);
    hit.objNrm = cross(e1, e2);
    //interpolate normal
    precise vec3 n1 = v1.normal - v0.normal;
    precise vec3 n2 = v2.normal - v0.normal;
    vec3 intNrm = v0.normal + fma(vec3(barys.x), n1, barys.y * n2);

    //model might have a different winding order than we expect
    // -> match sign of objNrm and intNrm
    hit.objNrm *= signBit(dot(hit.objNrm, intNrm));
    //normalize objNrm
    hit.objNrm = normalize(hit.objNrm);

    //translate from world to object space
    mat4x3 world2Obj = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
    hit.worldToObj = mat3(world2Obj);
    hit.objDir = normalize(mat3(world2Obj) * ray.direction);
    //check orientation
    // -> inward if direction and normal in opposite direction
    hit.inward = dot(hit.objDir, hit.objNrm) <= 0.0;

    //fetch object material
    hit.material = geom.material;
    //fetch material flags
    hit.flags = hit.inward ? hit.material.flagsInwards : hit.material.flagsOutwards;
    //light models are generally unaware of the scene's geometry and might have
    //sampled a light ray inside a geometry
    //-> test against and discard
    //address of expected ray medium
    uvec2 medium = hit.inward ? uvec2(hit.material.outside) : uvec2(hit.material.inside);
    if (ray.medium != medium)
        return ERROR_CODE_MEDIA_MISMATCH; 
    
    //translate from object to world space
    // hit.worldNrm = normalize(vec3(hit.objNrm * world2Obj));
    vec3 worldNrm = normalize(vec3(hit.objNrm * world2Obj));
    //create normal as seen by ray
    // float(bool) = bool ? 1.0 : 0.0
    // -> inward ? 1.0 : -1.0
    hit.rayNrm = worldNrm * (2.0 * float(hit.inward) - 1.0);
    
    //do matrix multiplication manually to improve error
    //See: https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/
    mat4x3 m = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
    hit.worldPos.x = m[3][0] + fma(m[0][0], hit.objPos.x, fma(m[1][0], hit.objPos.y, m[2][0] * hit.objPos.z));
    hit.worldPos.y = m[3][1] + fma(m[0][1], hit.objPos.x, fma(m[1][1], hit.objPos.y, m[2][1] * hit.objPos.z));
    hit.worldPos.z = m[3][2] + fma(m[0][2], hit.objPos.x, fma(m[1][2], hit.objPos.y, m[2][2] * hit.objPos.z));

    //done
    return RESULT_CODE_SUCCESS;
}

/**
 * Checks if observer and target are mutually visible.
*/
bool isVisible(vec3 observer, vec3 target) {
    //Direction and length of shadow ray
    vec3 dir = target - observer;
    float dist = length(dir);
    dir /= dist;

    //create and trace ray query
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF,
        observer,
        0.0, dir, dist
    );
    rayQueryProceedEXT(rayQuery);

    //points are mutable visible if no hit
    return rayQueryGetIntersectionTypeEXT(rayQuery, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT;
}

#endif
