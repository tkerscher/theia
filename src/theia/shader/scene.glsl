#ifndef _SCENE_INCLUDE
#define _SCENE_INCLUDE

#include "material.glsl"
#include "scatter.glsl"

layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer Vertex {
    vec3 position;
    vec3 normal;
};

// Indices of a triangle
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

bool traverseScene(inout Ray ray, float dist, rayQueryEXT query) {
    //NOTE: while query is a "in" parameter and thus copied it's actually sort
    //      of a address, thus query will hold all information also on the
    //      callers side after calling this function

    //sanitize direction
    ray.direction = normalize(ray.direction);

    //init query
    rayQueryInitializeEXT(
        query, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF, // mask -> hit anything
        ray.position,
        1e-5, //t_min; small epsilon to prevent hitting surfaces again after scatter
        ray.direction,
        max(1e-5,dist));
    //traverse scene
    rayQueryProceedEXT(query);

    //check if we hit anything
    bool hit = rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionTriangleEXT;
    if (hit) {
        //fetch actual distance travelled
        dist = rayQueryGetIntersectionTEXT(query, true);

        //if hit, the prob is to travel at least dist: p(x>=d)=exp(-mu*d)
        ray.log_prob -= ray.constants.mu_s * dist;
    }
    else {
        //if we did not hit anything the prop is to travel exactly dist p(d) = mu*exp(-mu*d)
        ray.log_prob += log(ray.constants.mu_s) - (ray.constants.mu_s * dist);
    }

    //update position
    ray.position += ray.direction * dist;
    //update transmission
    ray.log_trans -= ray.constants.mu_e * dist;
    //update ellapsed time
    ray.travelTime += dist / ray.constants.vg;

    //done -> return if we hit anything
    return hit;
}

#endif
