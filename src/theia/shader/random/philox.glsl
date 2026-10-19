#ifndef _PHILOX_INCLUDE
#define _PHILOX_INCLUDE

#include "random/util.glsl"

#ifndef PHILOX_ITERATION
#define PHILOX_ITERATION 10
#endif

uniform PhiloxParams {
    uvec2 key;
    uvec2 offset; //64 bit run index
} philoxParams;

uvec4 philox(uint stream, uint dim) {
    uvec4 state = uvec4(dim, stream, philoxParams.offset);
    uvec2 key = philoxParams.key;

    //create random number
    uint hi0, lo0, hi1, lo1;
    const uint M0 = 0xD2511F53u;
    const uint M1 = 0xCD9E8D57u;
    for (int i = 0; i < PHILOX_ITERATION; ++i) {
        //1 Round of philox
        umulExtended(M0, state.x, hi0, lo0);
        umulExtended(M1, state.z, hi1, lo1);
        state = uvec4(
            hi1 ^ state.y ^ key.x, lo1,
            hi0 ^ state.w ^ key.y, lo0
        );
        //bump key
        key += uvec2(0x9E3779B9u, 0xBB67AE85u);
    }
    
    return state;
}

//Every draw pulls a new fresh block and only uses the amount of numbers it
//needs. Philox is fine with skipping numbers so that causes no problem.
//Testing showed this approach to be faster than caching.

//random comes in four flavors as denoted by their suffix:
// - no suffix: returns random float in [0,1) and advances i by one
// - _s suffix: static version; i will stay the same
// - _b suffix: returns the binary 32bit word instead of float
// - _bs suffix: Combination of _b and _s
//Additionally, one can draw up to 4 numbers in one go (marked as 2D, 3D, 4D).
//They all advance i by only 1 regardless

uint random_bs(uint stream, uint dim) {
    return philox(stream, dim).x;
}
uvec2 random2D_bs(uint stream, uint dim) {
   return philox(stream, dim).xy; 
}
uvec3 random3D_bs(uint stream, uint dim) {
    return philox(stream, dim).xyz;
}
uvec4 random4D_bs(uint stream, uint dim) {
    return philox(stream, dim);
}

//the other functions are based on _bs and can be polyfilled
#include "random/rng_polyfill.glsl"

#endif
