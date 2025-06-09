#ifndef _INCLUDE_CALLBACK_STAT
#define _INCLUDE_CALLBACK_STAT

#include "ray.glsl"

layout(scalar) writeonly buffer Statistics {
    uint64_t created;
    uint64_t scattered;
    uint64_t hit;
    uint64_t detect;
    uint64_t volume;
    uint64_t lost;
    uint64_t decayed;
    uint64_t absorbed;
    uint64_t missed;
    uint64_t maxIter;
    uint64_t error;
    uint64_t mismatch;
} stats;

void onEvent(const RayState ray, ResultCode code, uint idx, uint i) {
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
    case RESULT_CODE_MAX_ITER:
        atomicAdd(stats.maxIter, 1);
        break;
    case ERROR_CODE_MEDIA_MISMATCH:
        atomicAdd(stats.mismatch, 1);
        break;
    }

    //collect all errors in one statistic
    if (code <= ERROR_CODE_MAX_VALUE) {
        atomicAdd(stats.error, 1);
    }
}

#endif
