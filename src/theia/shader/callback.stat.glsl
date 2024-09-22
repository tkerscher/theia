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
    uint missed;
    uint error;
} stats;

void onEvent(const RayState ray, ResultCode code, uint idx, uint i) {
    uint n;
    switch(code) {
    case RESULT_CODE_RAY_CREATED:
        atomicAdd(stats.created, 1);
        break;
    case RESULT_CODE_RAY_SCATTERED:
        atomicAdd(stats.scattered, 1);
        break;
    case RESULT_CODE_RAY_HIT:
        atomicAdd(stats.hit, 1);
        break;
    case RESULT_CODE_RAY_DETECTED:
        atomicAdd(stats.detect, 1);
        break;
    case RESULT_CODE_VOLUME_HIT:
        atomicAdd(stats.volume, 1);
        break;
    case RESULT_CODE_RAY_LOST:
        atomicAdd(stats.lost, 1);
        break;
    case RESULT_CODE_RAY_DECAYED:
        atomicAdd(stats.decayed, 1);
        break;
    case RESULT_CODE_RAY_ABSORBED:
        atomicAdd(stats.absorbed, 1);
        break;
    case RESULT_CODE_RAY_MISSED:
        atomicAdd(stats.missed, 1);
        break;
    default:
        //collect all errors in one statistic
        if (code <= ERROR_CODE_MAX_VALUE) {
            atomicAdd(stats.error, 1);
        }
        break;
    }
}

#endif
