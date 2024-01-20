#ifndef _INCLUDE_CALLBACK_STAT
#define _INCLUDE_CALLBACK_STAT

#include "ray.glsl"
#include "callback.type.glsl"

layout(scalar) writeonly buffer Statistics {
    uint scattered;
    uint hit;
    uint detect;
    uint volume;
    uint lost;
    uint decayed;
    uint absorbed;
} stats;

void onEvent(const Ray ray, EventType type, uint idx, uint i) {
    uint n;
    switch(type) {
    case EVENT_TYPE_RAY_SCATTERED:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.scattered, n);
        break;
    case EVENT_TYPE_RAY_HIT:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.hit, n);
        break;
    case EVENT_TYPE_RAY_DETECTED:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.detect, n);
        break;
    case EVENT_TYPE_VOLUME_CHANGED:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.volume, n);
        break;
    case EVENT_TYPE_RAY_LOST:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.lost, n);
        break;
    case EVENT_TYPE_RAY_DECAYED:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.decayed, n);
        break;
    case EVENT_TYPE_RAY_ABSORBED:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.absorbed, n);
        break;
    }
}

#endif
