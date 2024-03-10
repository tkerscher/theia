#ifndef _INCLUDE_RESPONSE_RECORD
#define _INCLUDE_RESPONSE_RECORD

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require

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

void response(HitItem item) {
    uint id = atomicAdd(hitQueueOut.count, 1);
    SAVE_HIT(item, hitQueueOut.queue, id)
}

#endif
