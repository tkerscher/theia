#ifndef _INCLUDE_RESPONSE_RECORD
#define _INCLUDE_RESPONSE_RECORD

#include "response.queue.glsl"

layout(scalar) writeonly buffer HitQueueOut {
    uint count;
    HitQueue queue;
} hitQueueOut;

// void response(HitItem item) {
//     uint n = subgroupAdd(1);
//     uint begin = 0;
//     if (subgroupElect()) {
//         begin = atomicAdd(hitQueueOut.count, n);
//     }
//     begin = subgroupBroadcastFirst(begin);
//     uint offset = subgroupExclusiveAdd(1);
//     SAVE_HIT(item, hitQueueOut.queue, begin + offset)
// }

void initResponse() {}

void response(HitItem item) {
    uint id = atomicAdd(hitQueueOut.count, 1);
    SAVE_HIT(item, hitQueueOut.queue, id)
}

void finalizeResponse() {}

#endif
