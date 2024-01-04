#ifndef _INCLUDE_RESPONSE_QUEUE
#define _INCLUDE_RESPONSE_QUEUE

#extension GL_EXT_control_flow_attributes : require

#include "response.common.glsl"

struct HitQueue {
    float posX[HIT_QUEUE_SIZE];
    float posY[HIT_QUEUE_SIZE];
    float posZ[HIT_QUEUE_SIZE];
    float dirX[HIT_QUEUE_SIZE];
    float dirY[HIT_QUEUE_SIZE];
    float dirZ[HIT_QUEUE_SIZE];
    float nrmX[HIT_QUEUE_SIZE];
    float nrmY[HIT_QUEUE_SIZE];
    float nrmZ[HIT_QUEUE_SIZE];

    float wavelength[HIT_QUEUE_SIZE];
    float time[HIT_QUEUE_SIZE];
    float contrib[HIT_QUEUE_SIZE];
};

//Thanks to GLSL not having references, but only passes variables by copy we
//would have to bet on the compiler being smart enough to inline the function
//and not copy around several MB of data, when we only want to read a few bytes.
//Since I don't like to bet, here are some ugly macros instead

#define LOAD_HIT(HIT, QUEUE, IDX) \
HitItem HIT = HitItem(\
    vec3(QUEUE.posX[IDX], QUEUE.posY[IDX], QUEUE.posZ[IDX]),\
    vec3(QUEUE.dirX[IDX], QUEUE.dirY[IDX], QUEUE.dirZ[IDX]),\
    vec3(QUEUE.nrmX[IDX], QUEUE.nrmY[IDX], QUEUE.nrmZ[IDX]),\
    QUEUE.wavelength[IDX], QUEUE.time[IDX], QUEUE.contrib[IDX]);

#define SAVE_HIT(HIT, QUEUE, IDX) \
QUEUE.posX[IDX] = HIT.position.x;\
QUEUE.posY[IDX] = HIT.position.y;\
QUEUE.posZ[IDX] = HIT.position.z;\
QUEUE.dirX[IDX] = HIT.direction.x;\
QUEUE.dirY[IDX] = HIT.direction.y;\
QUEUE.dirZ[IDX] = HIT.direction.z;\
QUEUE.nrmX[IDX] = HIT.normal.x;\
QUEUE.nrmY[IDX] = HIT.normal.y;\
QUEUE.nrmZ[IDX] = HIT.normal.z;\
QUEUE.wavelength[IDX] = HIT.wavelength;\
QUEUE.time[IDX] = HIT.time;\
QUEUE.contrib[IDX] = HIT.contrib;

#endif
