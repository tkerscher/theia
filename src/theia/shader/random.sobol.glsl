#ifndef _SOBOL_INCLUDE
#define _SOBOL_INCLUDE

#include "random.util.glsl"

uniform SobolParams {
    uint seed;
    uint offset;
} sobolParams;

/*
 * A shuffled, Owen-scrambled Sobol sampler, implemented with the
 * techniques from the paper "Practical Hash-based Owen Scrambling"
 * by Brent Burley, 2020, Journal of Computer Graphics Techniques.
 *
 * Note that unlike a standard high-dimensional Sobol sequence, this
 * Sobol sampler uses padding to achieve higher dimensions, as described
 * in Burley's paper.
 */

 //Based on the implementation by Blender Foundation (2011-2022) licensed under
 //Apache-2.0 and the supplementary code of the above paper

const uint sobolDirections[2][32] = {
    {
        0x80000000, 0x40000000, 0x20000000, 0x10000000,
        0x08000000, 0x04000000, 0x02000000, 0x01000000,
        0x00800000, 0x00400000, 0x00200000, 0x00100000,
        0x00080000, 0x00040000, 0x00020000, 0x00010000,
        0x00008000, 0x00004000, 0x00002000, 0x00001000,
        0x00000800, 0x00000400, 0x00000200, 0x00000100,
        0x00000080, 0x00000040, 0x00000020, 0x00000010,
        0x00000008, 0x00000004, 0x00000002, 0x00000001
    },
    {
        0x80000000, 0xC0000000, 0xA0000000, 0xF0000000,
        0x88000000, 0xCC000000, 0xAA000000, 0xFF000000,
        0x80800000, 0xC0C00000, 0xA0A00000, 0xF0F00000,
        0x88880000, 0xCCCC0000, 0xAAAA0000, 0xFFFF0000,
        0x80008000, 0xC000C000, 0xA000A000, 0xF000F000,
        0x88008800, 0xCC00CC00, 0xAA00AA00, 0xFF00FF00,
        0x80808080, 0xC0C0C0C0, 0xA0A0A0A0, 0xF0F0F0F0,
        0x88888888, 0xCCCCCCCC, 0xAAAAAAAA, 0xFFFFFFFF
    }/*,
    {
        0x80000000, 0xC0000000, 0x60000000, 0x90000000,
        0xE8000000, 0x5C000000, 0x8E000000, 0xC5000000,
        0x68800000, 0x9CC00000, 0xEE600000, 0x55900000,
        0x80680000, 0xC09C0000, 0x60EE0000, 0x90550000,
        0xE8808000, 0x5CC0C000, 0x8E606000, 0xC5909000,
        0x6868E800, 0x9C9C5C00, 0xEEEE8E00, 0x5555C500,
        0x8000E880, 0xC0005CC0, 0x60008E60, 0x9000C590,
        0xE8006868, 0x5C009C9C, 0x8E00EEEE, 0xC5005555
    },
    {
        0x80000000, 0xC0000000, 0x20000000, 0x50000000,
        0xF8000000, 0x74000000, 0xA2000000, 0x93000000,
        0xD8800000, 0x25400000, 0x59E00000, 0xE6D00000,
        0x78080000, 0xB40C0000, 0x82020000, 0xC3050000,
        0x208F8000, 0x51474000, 0xFBEA2000, 0x75D93000,
        0xA0858800, 0x914E5400, 0xDBE79E00, 0x25DB6D00,
        0x58800080, 0xE54000C0, 0x79E00020, 0xB6D00050,
        0x800800F8, 0xC00C0074, 0x200200A2, 0x50050093
    },
    {
        0x80000000, 0x40000000, 0x20000000, 0xB0000000,
        0xF8000000, 0xDC000000, 0x7A000000, 0x9D000000,
        0x5A800000, 0x2FC00000, 0xA1600000, 0xF0B00000,
        0xDA880000, 0x6FC40000, 0x81620000, 0x40BB0000,
        0x22878000, 0xB3C9C000, 0xFB65A000, 0xDDB2D000,
        0x78022800, 0x9C0B3C00, 0x5A0FB600, 0x2D0DDB00,
        0xA2878080, 0xF3C9C040, 0xDB65A020, 0x6DB2D0B0,
        0x800228F8, 0x400B3CDC, 0x200FB67A, 0xB00DDB9D
    }*/
};

/*
 * A fast, high-quality 32-bit mixing function.
 *
 * From https://github.com/skeeto/hash-prospector
 */
uint hp_mix32(uint n) {
    // The actual mixing function.
    n ^= n >> 16;
    n *= 0x21F0AAAD;
    n ^= n >> 15;
    n *= 0xD35A2D97;
    n ^= n >> 15;

    // Xor by a random number so input zero doesn't map to output zero.
    // The particular number used here isn't special.
    return n ^ 0xE6FE3BEB;
}

/*
 * Performs base-2 Owen scrambling on a reversed-bit integer.
 *
 * This is essentially the Laine-Karras permutation, but much higher
 * quality.  See https://psychopath.io/post/2021_01_30_building_a_better_lk_hash
 */
uint lk_hash(uint n, uint seed) {
  n ^= n * 0x3D20ADEA;
  n += seed;
  n *= (seed >> 16) | 1;
  n ^= n * 0x05526C56;
  n ^= n * 0x53A22864;

  return n;
}

//hash combine from Boost
uint combineHash(uint seed, uint v) {
    return seed ^ (v + 0x9E3779B9 + (seed << 6) + (seed >> 2));
}

/*
 * Calculates the Sobol bit pattern for the given dimension and index
*/
uint sobolBits(uint index, uint dim) {
    //fast path for dim 0. Because of padding this will be used most of the time
    //justifying the branching
    if (dim == 0) return bitfieldReverse(index);
    // if (dim > 4) return 0;

    //apply direction vectors
    uint result = 0;
    uint i = 0;
    while (index != 0) {
        uint j = findLSB(index); //count trailling zeros
        result ^= sobolDirections[dim][i + j];
        i += j + 1;
        //TODO: Blender source code mentions splitting this in two is necessary
        //      for x86, but we might do not need it
        index >>= j;
        index >>= 1;
    }
    return result;
}
//Owen scrambling
uint nestedScramble(uint x, uint seed) {
    x = bitfieldReverse(x);
    x = lk_hash(x, seed);
    x = bitfieldReverse(x);
    return x;
}
float scrambledSobol(uint scrambled_index, uint dim, uint seed) {
    uint bits = sobolBits(scrambled_index, dim);
    seed = combineHash(seed, dim);
    bits = nestedScramble(bits, seed);
    return normalizeUint(bits);
}

//theia random API

float random_s(uint a, uint dim) {
    //creates derivative seed using dimension to decorrelate dimensions
    uint seed = combineHash(sobolParams.seed, hp_mix32(dim));
    uint index = nestedScramble(a, seed); //scrambled index into Sobol sequence
    
    return scrambledSobol(index, 0, seed);
}
float random(uint a, inout uint dim) {
    float result = random_s(a, dim);
    dim += 1;
    return result;
}

vec2 random2D_s(uint a, uint dim) {
    //creates derivative seed using dimension to decorrelate dimensions
    uint seed = combineHash(sobolParams.seed, hp_mix32(dim));
    uint index = nestedScramble(a, seed); //scrambled index into Sobol sequence

    return vec2(
        scrambledSobol(index, 0, seed),
        scrambledSobol(index, 1, seed)
    );
}
vec2 random2D(uint a, inout uint dim) {
    vec2 result = random2D_s(a, dim);
    dim += 2;
    return result;
}

#endif
