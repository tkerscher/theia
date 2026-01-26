#ifndef _INCLUDE_UTIL_BUFFERS
#define _INCLUDE_UTIL_BUFFERS

layout(scalar, buffer_reference, buffer_reference_align=4) buffer FloatBuffer {
    float values[];
};

layout(scalar, buffer_reference, buffer_reference_align=4) buffer IntBuffer {
    int values[];
};

layout(scalar, buffer_reference, buffer_reference_align=4) buffer UIntBuffer {
    uint values[];
};

layout(scalar, buffer_reference, buffer_reference_align=4) buffer Counter {
    uint count;
};

uint incrementCounter(uvec2 counterAdr) {
    Counter counter = Counter(counterAdr);
    return atomicAdd(counter.count, 1);
}

uvec2 shiftAdr(uvec2 adr, uint offset) {
    uint carry;
    adr.x = uaddCarry(adr.x, offset, carry);
    adr.y += carry;
    return adr;
}

#endif
