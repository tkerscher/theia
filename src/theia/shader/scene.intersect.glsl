#ifndef _INCLUDE_SCENE_INTERSECT
#define _INCLUDE_SCENE_INTERSECT

#include "math.glsl"
#include "ray.glsl"
#include "result.glsl"
#include "scene.types.glsl"

//list of material used by each instanced geometry
//materials are referenced by their id in the material table
readonly buffer MaterialMap { uint materialMap[]; };
//Top level acceleration structure containing the scene
uniform accelerationStructureEXT tlas;

/*
 * process a given ray query and calculates the hit's position and surface
 * normal in both world and object space. Returns true, if successfull, false
 * otherwise.
 *
 * Note: Orientation of a triangle is determined based on its winding order.
 *       If viewed from the outside, a triangle must have counter-clockwise
 *       winding order, and vice versa.
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
    vec3 positions[3];
    rayQueryGetIntersectionTriangleVertexPositionsEXT(rayQuery, true, positions);
    vec2 barys = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    hit.customId = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    //reconstruct hit triangle
    precise vec3 e1 = positions[1] - positions[0];
    precise vec3 e2 = positions[2] - positions[0];
    hit.objPos = positions[0] + fma(vec3(barys.x), e1, barys.y * e2);
    //we can distinguish the sides of an triangle by the order of its vertices.
    //this is known as "winding order". By default we follow the standard used
    //in e.g. Blender or OpenGL and define the outward facing side to be
    //counter-clockwise
    #ifndef OUTWARD_FACE_CLOCK_WISE
    //default
    hit.objNrm = normalize(cross(e1, e2));
    #else
    //however, if for any reason we want the opposite behavior, we can just flip
    //the normal by flipping the cross product
    hit.objNrm = normalize(cross(e2, e1));
    #endif

    //translate from world to object space
    mat4x3 world2Obj = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
    hit.worldToObj = mat3(world2Obj);
    hit.objDir = normalize(mat3(world2Obj) * ray.direction);
    //check orientation
    // -> inward if direction and normal in opposite direction
    hit.inward = dot(hit.objDir, hit.objNrm) <= 0.0;

    //fetch object material
    int instanceId = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
    hit.materialIdx = materialMap[instanceId];
    //fetch material flags
    uint mediumIdx, flags;
    queryMaterialSide(hit.materialIdx, hit.inward, mediumIdx, flags);
    hit.flags = flags;
    //light models are generally unaware of the scene's geometry and might have
    //sampled a light ray inside a geometry
    //-> test against and discard
    //address of expected ray medium
    queryMaterialSide(hit.materialIdx, !hit.inward, mediumIdx, flags);
    if (ray.mediumIdx != mediumIdx)
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
