#ifndef _PHILOX_INCLUDE
#define _PHILOX_INCLUDE

// Philox 4x32 implementation

const int PHILOX_ITERATION = 10;

struct PhiloxState {
    uvec2 key;
    uvec4 counter;
    uvec4 buf;
    int idx;
} philoxState;

void philox_init(uvec2 key, uvec4 counter) {
    philoxState.key = key;
    philoxState.counter = counter;
    //philoxState.buf = uvec4(0,0,0,0);
    philoxState.idx = 3; //forces to calc on next rand()
}

float uintToFloat(uint src) {
    return uintBitsToFloat(0x3f800000u | (src & 0x7fffffu)) - 1.0;
}

float rand() {
    //bump buf
    philoxState.idx++;
    //buffered value (x already used)?
    if (philoxState.idx <= 3)
        return uintToFloat(philoxState.buf[philoxState.idx]);
    //reset idx
    philoxState.idx = 0;

    //increment counter
    uint carry = 1;
    uvec4 state = philoxState.counter;
    state.x = uaddCarry(state.x, 1, carry);
    state.y = uaddCarry(state.y, carry, carry);
    state.z = uaddCarry(state.z, carry, carry);
    state.w += carry; //we cant do anything more
    philoxState.counter = state;

    //ran out of buffered values -> calc next one
    const uint M0 = 0xD2511F53u;
    const uint M1 = 0xCD9E8D57u;
    uint hi0, lo0, hi1, lo1;
    uvec2 key = philoxState.key;
    for (int i = 0; i < PHILOX_ITERATION; ++i) {
        //1 Round of philox
        {
            umulExtended(M0, state.x, hi0, lo0);
            umulExtended(M1, state.z, hi1, lo1);
            state = uvec4(
                hi1^state.y^key.x, lo1,
                hi0^state.w^key.y, lo0
            );
        }
        //bump key
        {
            key.x += 0x9E3779B9u;
            key.y += 0xBB67AE85u;
        }
    }

    //update buffer
    philoxState.buf = state;

    //return first random number
    return uintToFloat(state.x);
}

#endif
