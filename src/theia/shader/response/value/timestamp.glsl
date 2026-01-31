#ifndef _INCLUDE_RESPONSE_STORE_TIME
#define _INCLUDE_RESPONSE_STORE_TIME

#include "util/buffers.glsl"

uniform ResponseParams {
    uvec2 queueAdr;
    uint queueSize;
} responseParams;

void response(HitItem item, uint idx, inout uint dim) {
    //assume value is detection probability
    // -> draw random number to decide whether to store the hit
    float value = responseValue(item, idx, dim);
    //compare strictly less to handle 0% chances correctly
    bool accept = random(idx, dim) < value;
    
    //store hit as timestamp
    if (accept) {
        uint i = incrementCounter(responseParams.queueAdr);
        uvec2 queueAdr = shiftAdr(responseParams.queueAdr, 4);

        FloatBuffer floats = FloatBuffer(queueAdr);
        floats.values[i] = item.time;

        #ifdef RESPONSE_STORE_OBJECT_ID
        queueAdr = shiftAdr(queueAdr, 4 * responseParams.queueSize);
        IntBuffer ints = IntBuffer(queueAdr);
        ints.values[i] = item.objectId;
        #endif
    }
}

#endif
