#ifndef _PHILOX_INCLUDE
#define _PHILOX_INCLUDE

#include "random/util.glsl"

#define PHILOX_ITERATION 10

uniform PhiloxParams {
    uvec2 key;
    uvec4 baseCount;
} philoxParams;

vec4 philoxBuffer = vec4(0.0);
uint philoxBufferIndex = 0xFFFFFFFFu;
//since we will divide dim by 4, we will never reach this value
//-> marks uninitialized buffer

uvec4 philox(uint stream, uint i) {
    //add index to base count
    uint carry = i; //reuse carry
    uvec4 state = philoxParams.baseCount;
    state.x = uaddCarry(state.x, carry, carry);
    state.y = uaddCarry(state.y, carry, carry);
    state.z = uaddCarry(state.z, carry, carry);
    state.w = uaddCarry(state.w, carry, carry);
    state.x += carry; //rollover
    //get stream
    uvec2 key = philoxParams.key;
    key.x = uaddCarry(key.x, stream, carry);
    key.y = uaddCarry(key.y, carry, carry);
    key.x += carry; //rollover

    //create random number
    uint hi0, lo0, hi1, lo1;
    const uint M0 = 0xD2511F53u;
    const uint M1 = 0xCD9E8D57u;
    for (int i = 0; i < PHILOX_ITERATION; ++i) {
        //1 Round of philox
        umulExtended(M0, state.x, hi0, lo0);
        umulExtended(M1, state.z, hi1, lo1);
        state = uvec4(
            hi1^state.y^key.x, lo1,
            hi0^state.w^key.y, lo0
        );
        //bump key
        key.x += 0x9E3779B9u;
        key.y += 0xBB67AE85u;
    }
    
    return state;
}

float random_s(uint stream, uint i) {
    //philox produces four numbers at once
    //-> use lower 2 bits as index into vec4
    uint idx = i & 0x03;
    i >>= 2;

    //check whether we need to update the buffer
    //since we expect most invocations to be at the same position inside their
    //streams, we use subgroup operations to skip divergent code paths.
    bool update = i != philoxBufferIndex;
    if (subgroupAll(update)) {
        philoxBuffer = normalizeUint(philox(stream, i));
    }
    else if (subgroupAny(update)) {
        if (update) philoxBuffer = normalizeUint(philox(stream, i));
    }

    //return value
    return philoxBuffer[idx];
}

vec2 random2D_s(uint stream, uint i) {
    return vec2(random_s(stream, i), random_s(stream, i+1));
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
