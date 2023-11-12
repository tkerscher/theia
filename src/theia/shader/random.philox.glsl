#ifndef _PHILOX_INCLUDE
#define _PHILOX_INCLUDE

#extension GL_KHR_shader_subgroup_vote : require

#define PHILOX_ITERATION 10

layout(scalar) uniform PhiloxParams {
    uvec2 key;
    uvec4 baseCount;
} philoxParams;
vec4 philoxBuffer = vec4(-1.0);
uint philoxBufferCount = 0;

void philox(uint stream, uint i) {
    //get counter (divide by 4)
    uint carry = i << 2; //reuse carry
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
    //convert uint to float
    philoxBuffer = uintBitsToFloat((state & 0x7fffffu) | 0x3f800000u) - 1.0;
}


float random(uint stream, uint i) {
    //get which element of buffer we need
    uint count = i << 2; //divide by 4
    uint idx = i & 0x03; //remainder

    //need to run philox?
    bool update = (
        philoxBuffer.x < 0.0 ||
        count != philoxBufferCount ||
        idx == 0
    );
    //check explicitly for subgroup uniformity as
    //it may result in more efficient code
    if (subgroupAll(update)) {
        philox(stream, i);
    }
    else if (subgroupAny(update)) {
        //mixed update flags
        if (update) philox(stream, i);
    }

    //return value
    return philoxBuffer[idx];
}

vec2 random2D(uint stream, uint i) {
    //TODO: Maybe skip one value if last in buffer, to do only one update?
    return vec2(random(stream, i), random(stream, i+1));
}

#endif
