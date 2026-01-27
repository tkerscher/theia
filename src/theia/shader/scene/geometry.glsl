#ifndef _INCLUDE_SCENE_GEOMETRY
#define _INCLUDE_SCENE_GEOMETRY

layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer Vertex {
    vec3 position;
    vec3 normal;
};

// Indices of a triangle
layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer Index {
    ivec3 idx;
};

#endif
