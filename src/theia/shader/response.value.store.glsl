#ifndef _INCLUDE_RESPONSE_STORE_VALUE
#define _INCLUDE_RESPONSE_STORE_VALUE

writeonly buffer ValueQueueOut {
    uint count;
    float value[VALUE_QUEUE_SIZE];
    float time[VALUE_QUEUE_SIZE];
    //int objectId[VALUE_QUEUE_SIZE];
} valueQueueOut;

void initResponse() {}

void response(HitItem item, uint idx, inout uint dim) {
    uint i = atomicAdd(valueQueueOut.count, 1);
    valueQueueOut.value[i] = responseValue(item, idx, dim);
    valueQueueOut.time[i] = item.time;
    //valueQueueOut.objectId[i] = item.objectId;
}

void finalizeResponse() {}

#endif
