#ifndef _SCENE_INCLUDE
#define _SCENE_INCLUDE

#include "material.glsl"

layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer Vertex {
    vec3 position;
    vec3 normal;
};

// Indices of a triangle
layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer Index {
    ivec3 idx;
};

layout(buffer_reference, scalar, buffer_reference_align=8) readonly buffer Geometry {
    Vertex vertices;    // &vertices[0]
    Index indices;      // &indices[0]
    Material material;
};

uniform accelerationStructureEXT tlas;

layout(scalar) uniform Scene {
    Geometry geometries;    // &geometries[0]
    Medium medium;          // medium of environment
} scene;

#endif