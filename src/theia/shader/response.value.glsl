#ifndef _INCLUDE_RESPONSE_VALUE
#define _INCLUDE_RESPONSE_VALUE

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require

layout(scalar) writeonly buffer ValueQueueOut {
    uint count;
    float value[VALUE_QUEUE_SIZE];
    float time[VALUE_QUEUE_SIZE];
} valueQueueOut;

void response(HitItem item) {
    //reserve space on the queue
    uint n = subgroupAdd(1);
    uint oldCount = 0;
    if (subgroupElect()) {
        oldCount = atomicAdd(valueQueueOut.count, n);
    }
    oldCount = subgroupBroadcastFirst(oldCount);

    //save value on queue
    uint idx = oldCount + subgroupExclusiveAdd(1);
    valueQueueOut.value[idx] = responseValue(item);
    valueQueueOut.time[idx] = item.time;
}

#endif
