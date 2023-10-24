#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_ray_query : require

layout(local_size_x = 4, local_size_y = 4) in;

#include "scene.glsl"

layout(rgba8) uniform image2D outImage;

layout(push_constant, scalar) uniform Push{
    vec2 dimension;
    vec3 position;
    vec3 direction;
    vec3 up;
    float maxDistance;
} push;

void main() {
    vec2 size = vec2(imageSize(outImage));
    vec2 pos = vec2(gl_GlobalInvocationID.xy);
    vec2 coord = pos / size - vec2(0.5);
    //flip y coord
    coord.y *= -1.0;
    coord *= push.dimension;

    //camera-worlds transformation
    vec3 dir = normalize(push.direction);
    vec3 up = normalize(push.up);
    vec3 third = cross(dir, up);
    vec3 position = push.position + coord.y * up + coord.x * third;

    //trace scene
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF, position, 0.0, dir, push.maxDistance);
    rayQueryProceedEXT(rayQuery);

    //calculate pixel color
    vec4 color;
    if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
        //hit

        //reconstruct triangle
        int geoIdx = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
        int trIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
        vec2 _bar = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
        vec3 bar = vec3(1.0 - _bar.x - _bar.y, _bar.x, _bar.y);
        //get vertices
        Geometry geom = geometries[geoIdx];
        ivec3 index = geom.indices[trIdx].idx;
        Vertex v0 = geom.vertices[index.x];
        Vertex v1 = geom.vertices[index.y];
        Vertex v2 = geom.vertices[index.z];
        //interpolate normal
        vec3 normal = v0.normal * bar.x + v1.normal * bar.y + v2.normal * bar.z;
        //normalize normal to (0,1)
        normal = (normal + 1) / 2;
        color = vec4(normal, 1.0);
    }
    else {
        //miss
        color = vec4(1.0,1.0,1.0,1.0);
    }
    imageStore(outImage, ivec2(gl_GlobalInvocationID.xy), color);
}
