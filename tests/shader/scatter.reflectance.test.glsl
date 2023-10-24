#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "scatter.glsl"

layout(local_size_x = 32) in;

struct Query {
    vec3 direction;
    vec3 normal;
    float wavelength;
};

layout(scalar) readonly buffer QueryBuffer{ Query q[]; };
layout(scalar) writeonly buffer ResultBuffer{ float r[]; };

layout(push_constant) uniform Scene {
    Material mat;
} scene;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    //check which side of material via normal (points outwards)
    float cos_i = dot(q[i].direction, q[i].normal);
    Medium med = cos_i <= 0.0 ? scene.mat.outside : scene.mat.inside;

    Ray ray = initRay(vec3(0.0), q[i].direction, q[i].wavelength, med);
    r[i] = reflectance(ray, scene.mat, q[i].normal);
}
