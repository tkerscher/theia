#ifndef _INCLUDE_CAMERARAYSOURCE_QUEUE
#define _INCLUDE_CAMERARAYSOURCE_QUEUE

#ifndef CAMERA_QUEUE_SIZE
#error "CAMERA_QUEUE_SIZE not defined"
#endif

#include "camera.common.glsl"

struct CameraQueue {
    float posX[CAMERA_QUEUE_SIZE];
    float posY[CAMERA_QUEUE_SIZE];
    float posZ[CAMERA_QUEUE_SIZE];
    float dirX[CAMERA_QUEUE_SIZE];
    float dirY[CAMERA_QUEUE_SIZE];
    float dirZ[CAMERA_QUEUE_SIZE];
    float contrib[CAMERA_QUEUE_SIZE];
    float timeDelta[CAMERA_QUEUE_SIZE];

    float hitPosX[CAMERA_QUEUE_SIZE];
    float hitPosY[CAMERA_QUEUE_SIZE];
    float hitPosZ[CAMERA_QUEUE_SIZE];
    float hitDirX[CAMERA_QUEUE_SIZE];
    float hitDirY[CAMERA_QUEUE_SIZE];
    float hitDirZ[CAMERA_QUEUE_SIZE];
    float hitNrmX[CAMERA_QUEUE_SIZE];
    float hitNrmY[CAMERA_QUEUE_SIZE];
    float hitNrmZ[CAMERA_QUEUE_SIZE];
};

//Thanks to GLSL not having references, but only passes variables by copy we
//would have to bet on the compiler being smart enough to inline the function
//and not copy around several MB of data, when we only want to read a few bytes.
//Since I don't like to bet, here are some ugly macros instead

#define LOAD_CAMERA(RAY, QUEUE, IDX) \
CameraRay RAY = CameraRay(\
    vec3(QUEUE.posX[IDX], QUEUE.posY[IDX], QUEUE.posZ[IDX]),\
    vec3(QUEUE.dirX[IDX], QUEUE.dirY[IDX], QUEUE.dirZ[IDX]),\
    QUEUE.contrib[IDX], QUEUE.timeDelta[IDX],\
    vec3(QUEUE.hitPosX[IDX], QUEUE.hitPosY[IDX], QUEUE.hitPosZ[IDX]),\
    vec3(QUEUE.hitDirX[IDX], QUEUE.hitDirY[IDX], QUEUE.hitDirZ[IDX]),\
    vec3(QUEUE.hitNrmX[IDX], QUEUE.hitNrmY[IDX], QUEUE.hitNrmZ[IDX]));

#define SAVE_CAMERA(RAY, QUEUE, IDX)\
QUEUE.posX[IDX] = RAY.position.x;\
QUEUE.posY[IDX] = RAY.position.y;\
QUEUE.posZ[IDX] = RAY.position.z;\
QUEUE.dirX[IDX] = RAY.direction.x;\
QUEUE.dirY[IDX] = RAY.direction.y;\
QUEUE.dirZ[IDX] = RAY.direction.z;\
QUEUE.contrib[IDX] = RAY.contrib;\
QUEUE.timeDelta[IDX] = RAY.timeDelta;\
QUEUE.hitPosX[IDX] = RAY.hitPosition.x;\
QUEUE.hitPosY[IDX] = RAY.hitPosition.y;\
QUEUE.hitPosZ[IDX] = RAY.hitPosition.z;\
QUEUE.hitDirX[IDX] = RAY.hitDirection.x;\
QUEUE.hitDirY[IDX] = RAY.hitDirection.y;\
QUEUE.hitDirZ[IDX] = RAY.hitDirection.z;\
QUEUE.hitNrmX[IDX] = RAY.hitNormal.x;\
QUEUE.hitNrmY[IDX] = RAY.hitNormal.y;\
QUEUE.hitNrmZ[IDX] = RAY.hitNormal.z;

#endif
