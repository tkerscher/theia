#ifndef _INCLUDE_RANDOM_RNG_POLYFILL
#define _INCLUDE_RANDOM_RNG_POLYFILL

#include "random/util.glsl"

//draw a random 32 bit word and advance dim by 1

uint random_b(uint stream, inout uint dim) {
    return random_bs(stream, dim++);
}
uvec2 random2D_b(uint stream, inout uint dim) {
   return random2D_bs(stream, dim++); 
}
uvec3 random3D_b(uint stream, inout uint dim) {
    return random3D_bs(stream, dim++);
}
uvec4 random4D_b(uint stream, inout uint dim) {
    return random4D_bs(stream, dim++);
}

//draw random float and keep dim const

float random_s(uint stream, uint dim) {
    return normalizeUint(random_bs(stream, dim));
}
vec2 random2D_s(uint stream, uint dim) {
    return normalizeUint(random2D_bs(stream, dim));
}
vec3 random3D_s(uint stream, uint dim) {
    return normalizeUint(random3D_bs(stream, dim));
}
vec4 random4D_s(uint stream, uint dim) {
    return normalizeUint(random4D_bs(stream, dim));
}

//draw random float and advance dim by 1

float random(uint stream, inout uint dim) {
    return normalizeUint(random_bs(stream, dim++));
}
vec2 random2D(uint stream, inout uint dim) {
    return normalizeUint(random2D_bs(stream, dim++));
}
vec3 random3D(uint stream, inout uint dim) {
    return normalizeUint(random3D_bs(stream, dim++));
}
vec4 random4D(uint stream, inout uint dim) {
    return normalizeUint(random4D_bs(stream, dim++));
}

#endif
