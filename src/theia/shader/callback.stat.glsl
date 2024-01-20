#ifndef _INCLUDE_CALLBACK_STAT
#define _INCLUDE_CALLBACK_STAT

#include "ray.glsl"

layout(scalar) writeonly buffer Statistics {
    uint created;
    uint scattered;
    uint hit;
    uint detect;
    uint volume;
    uint lost;
    uint decayed;
    uint absorbed;
    uint error;
} stats;

void onEvent(const Ray ray, ResultCode code, uint idx, uint i) {
    uint n;
    switch(code) {
    case RESULT_CODE_RAY_CREATED:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.created, n);
        break;
    case RESULT_CODE_RAY_SCATTERED:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.scattered, n);
        break;
    case RESULT_CODE_RAY_HIT:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.hit, n);
        break;
    case RESULT_CODE_RAY_DETECTED:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.detect, n);
        break;
    case RESULT_CODE_VOLUME_HIT:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.volume, n);
        break;
    case RESULT_CODE_RAY_LOST:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.lost, n);
        break;
    case RESULT_CODE_RAY_DECAYED:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.decayed, n);
        break;
    case RESULT_CODE_RAY_ABSORBED:
        n = subgroupAdd(1);
        if (subgroupElect()) atomicAdd(stats.absorbed, n);
        break;
    default:
        //collect all errors in one statistic
        if (code <= ERROR_CODE_MAX_VALUE) {
            n = subgroupAdd(1);
            if (subgroupElect()) atomicAdd(stats.error, n);
        }
        break;
    }
}

#endif
