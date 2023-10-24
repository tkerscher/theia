#extension GL_EXT_scalar_block_layout : require

// Philox 4x32 implementation
#define ITERATIONS 10

//x is stream -> increment key
//y is sample -> increment counter
layout(local_size_x = 32, local_size_y = 1) in;

//we always create 4 floats per invocation
writeonly buffer RNGBuffer{ vec4 u[]; };

layout(push_constant, scalar) uniform Push {
    uvec2 baseKey;
    uvec4 baseCounter;
    uint stride;
} push;

void main() {
    //build key
    uvec2 key = uvec2(0.0);
    key.x = uaddCarry(push.baseKey.x, gl_GlobalInvocationID.x, key.y);
    key.y += push.baseKey.y;
    //build state
    uvec4 state = uvec4(0.0);
    state.x = uaddCarry(push.baseCounter.x, gl_GlobalInvocationID.y, state.y);
    state.y = uaddCarry(push.baseCounter.y, state.y, state.z);
    state.z = uaddCarry(push.baseCounter.z, state.z, state.w);
    state.w += push.baseCounter.w;

    //generate 4 random floats
    const uint M0 = 0xD2511F53u;
    const uint M1 = 0xCD9E8D57u;
    uint hi0, lo0, hi1, lo1;
    for (int i = 0; i < ITERATIONS; ++i) {
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

    //save random numbers
    uint idx = gl_GlobalInvocationID.x * push.stride + gl_GlobalInvocationID.y;
    u[idx] = uintBitsToFloat(0x3f800000u | (state & 0x7fffffu)) - 1.0;
}
