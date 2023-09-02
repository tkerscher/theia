#version 460

#extension GL_GOOGLE_include_directive: require

#include "philox.glsl"

layout(local_size_x = 32) in;

layout(constant_id = 3) const int StreamSize = 10000; //10^4

//hard code key
layout(constant_id = 4) const uint keyLo = 0xDEADBEEF;
layout(constant_id = 5) const uint keyHi = 0xC00010FF;

layout(binding = 0) writeonly buffer outBuf{ float randOut[]; };

void main() {
    //give each stream its own key by adding its id
    uvec2 key = uvec2(keyHi, keyLo + gl_GlobalInvocationID.x);
    philox_init(key, uvec4(0,0,0,0));

    uint start = StreamSize * gl_GlobalInvocationID.x;
    uint end = start + StreamSize;

    for (uint n = start; n < end; n++) {
        randOut[n] = rand();
    }
}
