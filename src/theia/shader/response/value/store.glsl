#ifndef _INCLUDE_RESPONSE_STORE_VALUE
#define _INCLUDE_RESPONSE_STORE_VALUE

#include "util/buffers.glsl"

uniform ResponseParams {
    uvec2 queueAdr;
    uint queueSize;
} responseParams;

void response(const HitItem item, uint idx, inout uint dim) {
    uint i = incrementCounter(responseParams.queueAdr);
    uvec2 queueAdr = shiftAdr(responseParams.queueAdr, 4);
    //do not write out of bounds
    if (i >= responseParams.queueSize) return;

    FloatBuffer floats = FloatBuffer(queueAdr);
    floats.values[i] = responseValue(item, idx, dim); i += responseParams.queueSize;
    floats.values[i] = item.time; i += responseParams.queueSize;

    #ifdef RESPONSE_STORE_OBJECT_ID
    IntBuffer ints = IntBuffer(queueAdr);
    ints.values[i] = item.objectId;
    #endif
}

#endif
