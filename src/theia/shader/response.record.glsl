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

void response(HitItem item) {
    //count how many items we will be adding in this subgroup
    uint n = subgroupAdd(1);
    //elect one invocation to update counter in queue
    uint oldCount = 0;
    if (subgroupElect()) {
        oldCount = atomicAdd(hitQueueOut.count, n);
    }
    //only the first (i.e. elected) one has correct oldCount value -> broadcast
    oldCount = subgroupBroadcastFirst(oldCount);
    //now we effectevily reserved us the range [oldCount ... oldcount+n-1]

    //order the active invocations so each can write at their own spot
    uint id = subgroupExclusiveAdd(1);
    SAVE_HIT(item, hitQueueOut.queue, oldCount + id)
}

#endif
