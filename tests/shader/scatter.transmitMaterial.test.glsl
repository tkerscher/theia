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

struct Result {
    vec3 direction;
    float transmission;
    float refractive_index;
    float match_consts; 
};

layout(scalar) readonly buffer QueryBuffer{ Query q[]; };
layout(scalar) writeonly buffer ResultBuffer{ Result r[]; };

layout(push_constant) uniform Scene {
    Material mat;
} scene;

float checkMedium(const Ray ray) {
    MediumConstants consts = lookUpMedium(ray.medium, ray.wavelength);

    //check if values match
    if (abs(consts.n - ray.constants.n) >= 5e-7) { return -1.0; }
    if (abs(consts.vg - ray.constants.vg) >= 5e-7) { return -1.0; }
    if (abs(consts.mu_s - ray.constants.mu_s) >= 5e-7) { return -1.0; }
    if (abs(consts.mu_e - ray.constants.mu_e) >= 5e-7) { return -1.0; }
    //everything matches
    return 1.0;
}

void main() {
    uint i = gl_GlobalInvocationID.x;

    //check which side of material via normal (points outwards)
    float cos_i = dot(q[i].direction, q[i].normal);
    Medium med = cos_i <= 0.0 ? scene.mat.outside : scene.mat.inside;

    Ray ray = initRay(vec3(0.0), q[i].direction, q[i].wavelength, med);
    transmitMaterial(ray, scene.mat, q[i].normal);

    //save result
    r[i].direction = ray.direction;
    r[i].transmission = exp(ray.log_trans);
    r[i].refractive_index = ray.constants.n;
    //check here if the media and constants match
    //checking this in python would be unnecessary hard
    r[i].match_consts = checkMedium(ray);
}
