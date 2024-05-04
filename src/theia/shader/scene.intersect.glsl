#ifndef _INCLUDE_SCENE_INTERSECT
#define _INCLUDE_SCENE_INTERSECT

#include "math.glsl"
#include "ray.glsl"
#include "result.glsl"
#include "scene.glsl"

//util functions for scene intersections / ray queries

//Taken from Ray Tracing Gems: Chapter 6
// Normal points outward for rays exiting the surface, else is flipped.
vec3 offsetRay(vec3 p, vec3 n) {
    ivec3 of_i = ivec3(256.0 * n);
    
    vec3 p_i = vec3(
        intBitsToFloat(floatBitsToInt(p.x)+((p.x < 0.0) ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(p.y)+((p.y < 0.0) ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(p.z)+((p.z < 0.0) ? -of_i.z : of_i.z))
    );

    return vec3(
        abs(p.x) < (1.0 / 32.0) ? p.x+ (1.0/65536.0)*n.x : p_i.x,
        abs(p.y) < (1.0 / 32.0) ? p.y+ (1.0/65536.0)*n.y : p_i.y,
        abs(p.z) < (1.0 / 32.0) ? p.z+ (1.0/65536.0)*n.z : p_i.z
    );
}

/*
 * process a given ray query and calculates the hit's position and surface
 * normal in both world and object space. Returns true, if successfull, false
 * otherwise.
 *
 * Note: This version interpolates normals, which are known to violate energy
 *       conservation if used for BSDF.
 *
 * ray: Ray describing propagation state
 * rayQuery: Query used to trace scene
 * out mat: Material of intersected geometry
 * out inward: Orientation of ray relative to geometry
 * out objPos: Hit position in object space
 * out objNrm: Interpolated object normal in object space
 * out objDir: Ray direction in object space
 * out worldPos: Hit position in scene space
 * out worldNrm: Interpolated object normal in object space
 * out geomNormal: Object normal derived from mesh in scene space
*/
/*
ResultCode processRayQuery(
    const Ray ray, rayQueryEXT rayQuery,
    out Material mat, out bool inward,
    precise out vec3 objPos, precise out vec3 objNrm, out vec3 objDir,
    precise out vec3 worldPos, out vec3 worldNrm, out vec3 geomNormal
) {
    //check if we hit anything
    if (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionTriangleEXT)
        return ERROR_CODE_TRACE_ABORT;
      
    //fetch info about intersection
    int instanceId = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
    int triangleId = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    vec2 barys = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    //reconstruct hit triangle
    Geometry geom = geometries[instanceId];
    ivec3 index = geom.indices[triangleId].idx;
    Vertex v0 = geom.vertices[index.x];
    Vertex v1 = geom.vertices[index.y];
    Vertex v2 = geom.vertices[index.z];
    precise vec3 e1 = v1.position - v0.position;
    precise vec3 e2 = v2.position - v0.position;
    objPos = v0.position + fma(vec3(barys.x), e1, barys.y * e2);
    geomNormal = cross(e1, e2);
    //interpolate normal
    precise vec3 n1 = v1.normal - v0.normal;
    precise vec3 n2 = v2.normal - v0.normal;
    objNrm = v0.normal + fma(vec3(barys.x), n1, barys.y * n2);

    //model might have a different winding order than we expect
    // -> match sign of geomNormal and objNrm
    //(use +0.5 to avoid producing zero)
    geomNormal *= sign(sign(dot(objNrm, geomNormal)) + 0.5);

    //translate from world to object space
    mat4x3 world2Obj = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
    objDir = normalize(mat3(world2Obj) * ray.direction);
    //early direction test in object space using geomNormal for better numerical
    //precision. Don't use interpolated object normal (all user data is evil)
    inward = dot(objDir, geomNormal) <= 0.0; //normal and ray in opposite directions
    
    //light models are generally unaware of the scene's geometry and might have
    //sampled a light ray inside a geometry
    //-> test against and discard
    mat = geom.material;
    //address of expected ray medium
    uvec2 medium = inward ? uvec2(mat.outside) : uvec2(mat.inside);
    if (ray.medium != medium)
        return ERROR_CODE_MEDIA_MISSMATCH;

    //translate from object to world space
    worldNrm = normalize(vec3(objNrm * world2Obj));
    objNrm = normalize(objNrm);
    geomNormal = vec3(geomNormal * world2Obj);
    //ensure geo normals points in opposite general direction than ray
    geomNormal *= -sign(dot(geomNormal, ray.direction));
    geomNormal = normalize(geomNormal);

    //do matrix multiplication manually to improve error
    //See: https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/
    mat4x3 m = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
    worldPos.x = m[3][0] + fma(m[0][0], objPos.x, fma(m[1][0], objPos.y, m[2][0] * objPos.z));
    worldPos.y = m[3][1] + fma(m[0][1], objPos.x, fma(m[1][1], objPos.y, m[2][1] * objPos.z));
    worldPos.z = m[3][2] + fma(m[0][2], objPos.x, fma(m[1][2], objPos.y, m[2][2] * objPos.z));

    //done
    return RESULT_CODE_SUCCESS;
}
*/

/**
 * Structure describing the intersection of a ray query with geometry.
 * Includes information about hit both in world space (where the ray lives)
 * and object space (where geometry is defined), i.e. before transformation
 * applied.
*/
struct SurfaceHit {
    bool valid;             ///< True, if an actual hit happened

    //Geometries mark a boundary between an inside and an outside volume.
    //Their normals distinguish them by pointing outwards.
    //Materials finally assign both volums a medium.

    Material material;      ///< Material of the intersected geometry
    bool inward;            ///< Direction of ray respective to geometry
    int customId;           ///< Custom ID of the intersected geometry
    uint flags;             ///< Material flags for specific direction (inward)

    //Variables defined in world space, i.e. the same as the ray

    precise vec3 worldPos;  ///< Hit position in world space
    vec3 worldNrm;          ///< Geometry normal at hit position in world space
    vec3 rayNrm;            ///< surface normal opposing ray direction

    //The following variables are define object space, i.e. geometry's
    //coordinates before any transformation were applied

    precise vec3 objPos;    ///< Hit position in object space
    precise vec3 objNrm;    ///< Geometry normal at hit position in object space
    vec3 objDir;            ///< Ray direction in object space
};

/*
 * process a given ray query and calculates the hit's position and surface
 * normal in both world and object space. Returns true, if successfull, false
 * otherwise.
 *
 * Note: Normals always points outwards.
 *
 * Note: This version does not interpolate normals, but use it to determine the
 *       orientation of the material (inward/outward).
 *
 * ray: Ray describing propagation state
 * rayQuery: Query used to trace scene
 * out hit: Structure describing the hit
*/
ResultCode processRayQuery(
    const Ray ray, rayQueryEXT rayQuery, out SurfaceHit hit
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
        return ERROR_CODE_MEDIA_MISSMATCH; 
    
    //translate from object to world space
    hit.worldNrm = normalize(vec3(hit.objNrm * world2Obj));
    //create normal as seen by ray
    // float(bool) = bool ? 1.0 : 0.0
    // -> inward ? 1.0 : -1.0
    hit.rayNrm = hit.worldNrm * (2.0 * float(hit.inward) - 1.0);
    
    //do matrix multiplication manually to improve error
    //See: https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/
    mat4x3 m = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
    hit.worldPos.x = m[3][0] + fma(m[0][0], hit.objPos.x, fma(m[1][0], hit.objPos.y, m[2][0] * hit.objPos.z));
    hit.worldPos.y = m[3][1] + fma(m[0][1], hit.objPos.x, fma(m[1][1], hit.objPos.y, m[2][1] * hit.objPos.z));
    hit.worldPos.z = m[3][2] + fma(m[0][2], hit.objPos.x, fma(m[1][2], hit.objPos.y, m[2][2] * hit.objPos.z));

    //done
    return RESULT_CODE_SUCCESS;
}

#endif
