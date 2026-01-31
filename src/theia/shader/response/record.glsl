#ifndef _INCLUDE_RESPONSE_RECORD
#define _INCLUDE_RESPONSE_RECORD

uniform ResponseParams {
    uvec2 queueAdr;
    uint queueSize;
} responseParams;

void response(HitItem item, uint idx, inout uint dim) {
    uint id = incrementCounter(responseParams.queueAdr);
    uvec2 queueAdr = shiftAdr(responseParams.queueAdr, 4);
    if (id < responseParams.queueSize) {
        saveHitItem(queueAdr, responseParams.queueSize, id, item);
    }
}

#endif
