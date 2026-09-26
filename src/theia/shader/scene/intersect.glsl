#ifndef _INCLUDE_SCENE_INTERSECT
#define _INCLUDE_SCENE_INTERSECT

#include "util/float.glsl"

//constants for intersection error calculation. can be overwritten to match
//different hardware
//
//errors in the barycentric coordinates. Depends on the hardware implementation.
//A likely candidate is the algorithm by Woop et al., which reports 5 epsilon
//error. Since we get another factor 2 from the extent estimate the following
//3 epsilon are actually 6.
//This should match NVIDIA hardware but other vendor might be off worse.
//Change this variable in that case.
#ifndef RAY_INTERSECT_BARYS_EPS
#define RAY_INTERSECT_BARYS_EPS 1.7881393432617188e-7
#endif
//the following define the errors occuring when transforming position from
//object space to world space (3 and 2 epsilon respectively)
#define RAY_OBJ2WORLD_MATMUL_EPS 1.7881393432617188e-7
#define RAY_OBJ2WORLD_TRANS_EPS  1.1920928955078125e-7
//this defines the error occuring when transforming the ray into object space
//done by the ray tracing hardware to prepare tracing the ray in object space
#ifndef RAY_WORLD2OBJ_EPS
#define RAY_WORLD2OBJ_EPS 1.1920928955078125e-7
#endif

//list of material used by each instanced geometry
//materials are referenced by their id in the material table
readonly buffer MaterialMap { uint materialMap[]; };

#ifndef POSITION_FETCH_ENABLED
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
    vec3 x0 = positions[0];
    #undef positions

    #else

    //we have to manually fetch vertex positions
    //start by fetching memory addresses of vertex and index buffer of this geometry
    uvec4 address = geometryMap[gl_InstanceID];
    Vertex vertices = Vertex(address.xy);
    Index indices = Index(address.zw);
    //fetch indices of hit triangle
    ivec3 index = indices[gl_PrimitiveID].idx;
    Vertex v0 = vertices[index.x];
    Vertex v1 = vertices[index.y];
    Vertex v2 = vertices[index.z];
    vec3 x0 = v0.position;
    //calculate edges
    precise vec3 e1 = v1.position - x0;
    precise vec3 e2 = v2.position - x0;

    #endif

    //reconstruct hit position
    hit.objPos = x0 + fmaKHR(vec3(barys.x), e1, barys.y * e2);

    //we can distinguish the sides of an triangle by the order of its vertices.
    //this is known as "winding order". By default we follow the standard used
    //in e.g. Blender or OpenGL and define the outward facing side to be
    //counter-clockwise
    #ifndef OUTWARD_FACE_CLOCK_WISE
    //default
    vec3 objNrm = cross(e1, e2);
    #else
    //however, if for any reason we want the opposite behavior, we can just flip
    //the normal by flipping the cross product
    vec3 objNrm = cross(e2, e1);
    #endif
    hit.objNrm = normalize(objNrm);

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
    //we keep the normals unnormalized for now. this makes the error calculation math easier
    vec3 worldNrm = vec3(objNrm * gl_WorldToObjectEXT);
    float worldScale = inversesqrt(dot(worldNrm, worldNrm));
    worldNrm *= worldScale; //normalize
    //create normal as seen by ray
    hit.rayNrm = hit.inward ? worldNrm : -worldNrm;

    //do matrix multiplication manually to improve error
    //See: https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/
    mat4x3 o2w = gl_ObjectToWorldEXT;
    hit.worldPos.x = o2w[3][0] + fmaKHR(o2w[0][0], hit.objPos.x, fmaKHR(o2w[1][0], hit.objPos.y, o2w[2][0] * hit.objPos.z));
    hit.worldPos.y = o2w[3][1] + fmaKHR(o2w[0][1], hit.objPos.x, fmaKHR(o2w[1][1], hit.objPos.y, o2w[2][1] * hit.objPos.z));
    hit.worldPos.z = o2w[3][2] + fmaKHR(o2w[0][2], hit.objPos.x, fmaKHR(o2w[1][2], hit.objPos.y, o2w[2][2] * hit.objPos.z));

    //error calculation to determine minimal ray offset to prevent self-intersection
    //adapted from https://github.com/NVIDIA/self-intersection-avoidance/
    
    //upper error bound on reconstructed object space intersection
    vec3 ext3 = abs(e1) + abs(e2) + abs(e1 - e2);
    float ext = max(max(ext3.x, ext3.y), ext3.z);
    vec3 objErr = fmaKHR(vec3(FLT_U), abs(x0), vec3(RAY_INTERSECT_BARYS_EPS * ext));
    //upper error bound on world intersection bound caused by trafo
    mat4x3 abs_o2w = mat4x3(abs(o2w[0]), abs(o2w[1]), abs(o2w[2]), abs(o2w[3]));
    vec3 worldErr = fmaKHR(
        vec3(RAY_OBJ2WORLD_MATMUL_EPS),
        mat3(abs_o2w) * abs(hit.objPos),
        (RAY_OBJ2WORLD_TRANS_EPS * abs(o2w[3]))
    );
    //error from world to object trafo (next tracing)
    mat4x3 w2o = gl_WorldToObjectEXT;
    mat4x3 abs_w2o = mat4x3(abs(w2o[0]), abs(w2o[1]), abs(w2o[2]), abs(w2o[3]));
    objErr = fmaKHR(vec3(RAY_WORLD2OBJ_EPS), (abs_w2o * vec4(abs(hit.worldPos), 1.0)), objErr);
    //project errors to normals to get offsets
    float worldOffset = dot(worldErr, abs(worldNrm));
    float objOffest = dot(objErr, abs(objNrm)); //!!! unnormalized objNrm on purpose !!!
    //combine offsets
    hit.rayOffset = fmaKHR(worldScale, objOffest, worldOffset);

    //done
    return RESULT_CODE_SUCCESS;
}

#endif
