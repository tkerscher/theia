#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define USE_GLOBAL_MEDIUM 1
#include "scatter.surface.glsl"

layout(local_size_x = 32) in;

struct Query {
    vec3 direction;
    vec3 normal;
    float wavelength;
};

layout(scalar) readonly buffer QueryBuffer{ Query q[]; };
layout(scalar) writeonly buffer ResultBuffer{ vec3 r[]; };

layout(push_constant) uniform Scene {
    uvec2 mat;
} scene;

void main() {
    uint i = gl_GlobalInvocationID.x;
    Material mat = Material(scene.mat);
    //check which side of material via normal (points outwards)
    float cos_i = dot(q[i].direction, q[i].normal);
    bool inward = cos_i <= 0.0;
    vec3 normal = inward ? q[i].normal : -q[i].normal;
    Medium med = inward ? mat.outside : mat.inside;
    //look up refractive index
    MediumConstants consts = lookUpMedium(med, q[i].wavelength);

    //assemble dummy ray and hit
    RayState ray = RayState(
        vec3(0.0), q[i].direction,
        q[i].wavelength, 0.0, 1.0, 0.0,
        consts
    );
    SurfaceHit hit = SurfaceHit(
        true, mat, inward,
        0, 0,
        vec3(0.0), normal,
        vec3(0.0), q[i].normal, q[i].direction,
        mat3(1.0)
    );

    //calculate reflectance
    Reflectance s = fresnelReflect(ray, hit);
    r[i] = vec3(s.r_s, s.r_p, s.n_tr);
}
