#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_ray_query : require

#include "scene.glsl"
#include "philox.glsl"

layout(local_size_x = 32) in;

struct Query {
    vec3 position;
    vec3 direction;
    float log_trans;
    float log_prob;
    float t0;
};

struct Result {
    vec3 position;
    float log_trans;
    float log_prob;
    float t;
    float hit;
};

layout(scalar) readonly buffer QueryBuffer{ Query query[]; };
layout(scalar) writeonly buffer ResultBuffer{ Result result[]; };

layout(scalar) buffer Debug{ uint64_t d[]; };

layout(scalar, push_constant) uniform Push{
    Medium medium;
    //float wavelength;
} push;

void main() {
    uint i = gl_GlobalInvocationID.x;
    Query q = query[i];

    if (i == 0) {
        d[0] = uint64_t(push.medium);
        d[1] = uint64_t(push.medium.mu_s);
        d[2] = uint64_t(0);
    }

    //init ray
    Ray ray = initRay(
        q.position,
        q.direction,
        500.0, //push.wavelength, //for whatever reason adding the wavelength corrups the medium pointer !?
        push.medium,
        q.log_trans,
        q.t0);
    ray.log_prob = q.log_prob;
    
    //sample distance
    float dist = -log(1.0 - rand()) / ray.constants.mu_s;

    //traverse
    rayQueryEXT rayQuery;
    bool hit = traverseScene(ray, dist, rayQuery);

    //save result
    result[i] = Result(
        ray.position,
        ray.log_trans,
        ray.log_prob,
        ray.travelTime,
        hit ? 1.0 : -1.0
    );
}
