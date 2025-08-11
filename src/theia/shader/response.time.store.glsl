#ifndef _INCLUDE_RESPONSE_STORE_TIME
#define _INCLUDE_RESPONSE_STORE_TIME

writeonly buffer ValueQueueOut {
    uint count;
    float time[VALUE_QUEUE_SIZE];
    #ifdef RESPONSE_STORE_OBJECT_ID
    int objectId[VALUE_QUEUE_SIZE];
    #endif
} valueQueueOut;

void initResponse() {}

void response(HitItem hit, uint idx, inout uint dim) {
    //assume value is detection probability
    //-> draw a random number to decide wether to store the hit
    float value = responseValue(hit, idx, dim);
    if (random(idx, dim) < value) {
        //accept hit
        uint q_idx = atomicAdd(valueQueueOut.count, 1);
        valueQueueOut.time[q_idx] = hit.time;
        #ifdef RESPONSE_STORE_OBJECT_ID
        valueQueueOut.objectId[q_idx] = hit.objectId;
        #endif
    }
}

void finalizeResponse() {}

#endif
