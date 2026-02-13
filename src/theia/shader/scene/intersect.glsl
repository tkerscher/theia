#ifndef _INCLUDE_SCENE_INTERSECT
#define _INCLUDE_SCENE_INTERSECT

//list of material used by each instanced geometry
//materials are referenced by their id in the material table
readonly buffer MaterialMap { uint materialMap[]; };

#ifndef USE_RAY_TRACING_POSITION_FETCH
//unfortunately, fetching vertex position from tlas is an optional feature
//and in this case it's not available, so we have to do it ourselves...

#include "scene/geometry.glsl"

//a bit lazy but it'll do:
// - addresses[i].xy -> address of vertices of i-th instance
// - addresses[i].zw -> address of indices of i-th instance
readonly buffer GeometryMap { uvec4 geometryMap[]; };

#endif

/**
 * Usable inside a closest hit shader to resolve the intersection and storing
 * all its information in a SurfaceHit struct. Returns RESULT_CODE_SUCCESS if
 * successful or an error code otherwise.
*/
ResultCode resolveIntersection(
    uint rayMediumIdx,      ///< Index of the ray's current medium
    vec2 barys,             ///< barycentric coordinates of hit
    out SurfaceHit hit      ///< Resolved intersection
) {
    //fetch hit triangle
    #ifdef POSITION_FETCH_ENABLED

    #define positions gl_HitTriangleVertexPositionsEXT
    precise vec3 e1 = positions[1] - positions[0];
    precise vec3 e2 = positions[2] - positions[0];

    #else

    //we have to manually fetch vertex positions
    //start by fetching memory addresses of vertex and index buffer of this geometry
    uvec4 address = geometryMap[gl_InstanceID];
    Vertex vertices = Vertex(address.xy);
    Index indices = Index(address.wz);
    //fetch indices of hit triangle
    ivec3 index = indices[gl_PrimitiveID].idx;
    Vertex v0 = vertices[index.x];
    Vertex v1 = vertices[index.y];
    Vertex v2 = vertices[index.z];
    //calculate edges
    precise vec3 e1 = v1.position - v0.position;
    precise vec3 e2 = v2.position - v0.position;

    #endif

    //reconstruct hit position
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
    hit.worldToObj = mat3(gl_WorldToObjectEXT);
    hit.objDir = gl_ObjectRayDirectionEXT;
    //check orientation
    // -> inward if direction and normal in opposite direction
    hit.inward = dot(hit.objDir, hit.objNrm) <= 0.0;

    //fetch object material
    hit.materialIdx = uint(gl_InstanceCustomIndexEXT);
    //fetch material flags
    uint mediumIdx, flags;
    queryMaterialSide(hit.materialIdx, hit.inward, mediumIdx, flags);
    hit.otherMediumIdx = mediumIdx;
    hit.flags = flags;

    //Sanity check whether the ray actually comes from the expected medium
    queryMaterialSide(hit.materialIdx, !hit.inward, mediumIdx, flags);
    bool checkMismatch = (hit.flags & MATERIAL_SKIP_MISMATCH_TEST_BIT) == 0; //check if not set
    if (checkMismatch && rayMediumIdx != mediumIdx)
        return ERROR_CODE_MEDIA_MISMATCH;
    
    //translate from object to world space
    vec3 worldNrm = normalize(vec3(hit.objNrm * gl_WorldToObjectEXT));
    //create normal as seen by ray
    // float(bool) = bool ? 1.0 : 0.0
    // -> inward ? 1.0 : -1.0
    hit.rayNrm = worldNrm * (2.0 * float(hit.inward) - 1.0);

    //do matrix multiplication manually to improve error
    //See: https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/
    mat4x3 m = gl_ObjectToWorldEXT;
    hit.worldPos.x = m[3][0] + fma(m[0][0], hit.objPos.x, fma(m[1][0], hit.objPos.y, m[2][0] * hit.objPos.z));
    hit.worldPos.y = m[3][1] + fma(m[0][1], hit.objPos.x, fma(m[1][1], hit.objPos.y, m[2][1] * hit.objPos.z));
    hit.worldPos.z = m[3][2] + fma(m[0][2], hit.objPos.x, fma(m[1][2], hit.objPos.y, m[2][2] * hit.objPos.z));

    //done
    return RESULT_CODE_SUCCESS;
}

#endif
