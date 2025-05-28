#ifndef _INCLUDE_RESPONSE_STORE_VALUE
#define _INCLUDE_RESPONSE_STORE_VALUE

layout(scalar) writeonly buffer ValueQueueOut {
    uint count;
    float value[VALUE_QUEUE_SIZE];
    float time[VALUE_QUEUE_SIZE];
} valueQueueOut;

void initResponse() {}

void response(HitItem item, uint idx, inout uint dim) {
    // uint idx = atomicAdd(valueQueueOut.count, 1);
    valueQueueOut.value[idx] = responseValue(item, idx, dim);
    valueQueueOut.time[idx] = item.time;
}

void finalizeResponse() {}

#endif
