#ifndef _SOBOL_INCLUDE
#define _SOBOL_INCLUDE

#extension GL_KHR_shader_subgroup_vote : require

#include "random.util.glsl"

//Based on the implementation presented in PBRT v4:
//M. Pharr, W. Jakob, G. Humphreys: Physical Based Rendering, 4th Edition. 2023

#ifndef _SOBOL_NO_SCRAMBLE

#include "hash.glsl"

uint fastOwenScramble(uint v, uint seed) {
    v = bitfieldReverse(v);
    v ^= v * 0x3D20ADEA;
    v += seed;
    v *= (seed >> 16) | 1;
    v ^= v * 0x05526C56;
    v ^= v * 0x53A22864;
    return bitfieldReverse(v);
}

#endif

layout(scalar) readonly buffer SobolMatrices {
    //make the number of generator matrices dynamic. allows to truncate it
    uint sobolMatrices[];
};

layout(scalar) uniform SobolParams {
    uint seed;
    uint offset;
} sobolParams;

//draws a-th scrambled sobol sample for dim-th dimension
float random_s(uint a, uint dim) {
    uint v = 0;
    //apply offset
    a += sobolParams.offset;

    // for (uint i = dim * 32; a != 0; a >>= 1, ++i) {
    //     if (a & 1)
    //         v ^= sobolMatrices[i];
    // }
    uint i = dim * 32;
    while (subgroupAny(a != 0)) {
        if ((a & 1) != 0)
            v ^= sobolMatrices[i];
        
        a >>= 1;
        i++;
    }

    //scramble
#ifndef _SOBOL_NO_SCRAMBLE
    //owen scrambling requires random seeds
    //ensure randomness via a hash
    uint h = hash(dim, sobolParams.seed);
    v = fastOwenScramble(v, h);
#endif

    return normalizeUint(v);
}

vec2 random2D_s(uint a, uint dim) {
    return vec2(random_s(a, dim), random_s(a, dim + 1));
}

float random(uint stream, inout uint i) {
    float result = random_s(stream, i);
    i += 1;
    return result;
}

vec2 random2D(uint stream, inout uint i) {
    vec2 result = random2D_s(stream, i);
    i += 2;
    return result;
}

#endif
