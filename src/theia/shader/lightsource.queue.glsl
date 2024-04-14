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
    float wavelength[LIGHT_QUEUE_SIZE];
    float startTime[LIGHT_QUEUE_SIZE];
    float contrib[LIGHT_QUEUE_SIZE];
};

//Thanks to GLSL not having references, but only passes variables by copy we
//would have to bet on the compiler being smart enough to inline the function
//and not copy around several MB of data, when we only want to read a few bytes.
//Since I don't like to bet, here are some ugly macros instead

#define LOAD_SAMPLE(RAY, QUEUE, IDX) \
SourceRay RAY = SourceRay(\
    vec3(QUEUE.posX[IDX], QUEUE.posY[IDX], QUEUE.posZ[IDX]),\
    vec3(QUEUE.dirX[IDX], QUEUE.dirY[IDX], QUEUE.dirZ[IDX]),\
    QUEUE.wavelength[IDX],\
    QUEUE.startTime[IDX],\
    QUEUE.contrib[IDX]);

#define SAVE_SAMPLE(RAY, QUEUE, IDX) \
QUEUE.posX[IDX] = RAY.position.x;\
QUEUE.posY[IDX] = RAY.position.y;\
QUEUE.posZ[IDX] = RAY.position.z;\
QUEUE.dirX[IDX] = RAY.direction.x;\
QUEUE.dirY[IDX] = RAY.direction.y;\
QUEUE.dirZ[IDX] = RAY.direction.z;\
QUEUE.wavelength[IDX] = RAY.wavelength;\
QUEUE.startTime[IDX] = RAY.startTime;\
QUEUE.contrib[IDX] = RAY.contrib;

#endif
