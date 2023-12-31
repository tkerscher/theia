#ifndef _INCLUDE_LIGHTSOURCE_QUEUE
#define _INCLUDE_LIGHTSOURCE_QUEUE

#extension GL_EXT_control_flow_attributes : require

#ifndef LIGHT_QUEUE_SIZE
#error "LIGHT_QUEUE_SIZE not defined"
#endif

#include "lightsource.common.glsl"

struct LightSourceQueue {
    //ray
    float posX[LIGHT_QUEUE_SIZE];
    float posY[LIGHT_QUEUE_SIZE];
    float posZ[LIGHT_QUEUE_SIZE];
    float dirX[LIGHT_QUEUE_SIZE];
    float dirY[LIGHT_QUEUE_SIZE];
    float dirZ[LIGHT_QUEUE_SIZE];
    //sample
    float wavelength[N_LAMBDA][LIGHT_QUEUE_SIZE];
    float startTime[N_LAMBDA][LIGHT_QUEUE_SIZE];
    float contrib[N_LAMBDA][LIGHT_QUEUE_SIZE];
};

//Thanks to GLSL not having references, but only passes variables by copy we
//would have to bet on the compiler being smart enough to inline the function
//and not copy around several MB of data, when we only want to read a few bytes.
//Since I don't like to bet, here are some ugly macros instead

#define LOAD_SAMPLE(RAY, QUEUE, IDX) \
SourceSample RAY##_samples[N_LAMBDA];\
[[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {\
    RAY##_samples[i] = SourceSample(\
        QUEUE.wavelength[i][IDX],\
        QUEUE.startTime[i][IDX],\
        QUEUE.contrib[i][IDX]);\
}\
SourceRay RAY = SourceRay(\
    vec3(QUEUE.posX[idx], QUEUE.posY[idx], QUEUE.posZ[idx]),\
    vec3(QUEUE.dirX[idx], QUEUE.dirY[idx], QUEUE.dirZ[idx]),\
    RAY##_samples);

#define SAVE_SAMPLE(RAY, QUEUE, IDX) \
[[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {\
    QUEUE.wavelength[i][idx] = RAY.samples[i].wavelength;\
    QUEUE.startTime[i][idx] = RAY.samples[i].startTime;\
    QUEUE.contrib[i][idx] = RAY.samples[i].contrib;\
}\
QUEUE.posX[idx] = RAY.position.x;\
QUEUE.posY[idx] = RAY.position.y;\
QUEUE.posZ[idx] = RAY.position.z;\
QUEUE.dirX[idx] = RAY.direction.x;\
QUEUE.dirY[idx] = RAY.direction.y;\
QUEUE.dirZ[idx] = RAY.direction.z;

#endif