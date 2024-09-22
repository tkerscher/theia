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

    #ifdef CAMERA_QUEUE_POLARIZED
    float polX[CAMERA_QUEUE_SIZE];
    float polY[CAMERA_QUEUE_SIZE];
    float polZ[CAMERA_QUEUE_SIZE];

    float m00[CAMERA_QUEUE_SIZE];
    float m01[CAMERA_QUEUE_SIZE];
    float m02[CAMERA_QUEUE_SIZE];
    float m03[CAMERA_QUEUE_SIZE];
    float m10[CAMERA_QUEUE_SIZE];
    float m11[CAMERA_QUEUE_SIZE];
    float m12[CAMERA_QUEUE_SIZE];
    float m13[CAMERA_QUEUE_SIZE];
    float m20[CAMERA_QUEUE_SIZE];
    float m21[CAMERA_QUEUE_SIZE];
    float m22[CAMERA_QUEUE_SIZE];
    float m23[CAMERA_QUEUE_SIZE];
    float m30[CAMERA_QUEUE_SIZE];
    float m31[CAMERA_QUEUE_SIZE];
    float m32[CAMERA_QUEUE_SIZE];
    float m33[CAMERA_QUEUE_SIZE];

    float hitPolX[CAMERA_QUEUE_SIZE];
    float hitPolY[CAMERA_QUEUE_SIZE];
    float hitPolZ[CAMERA_QUEUE_SIZE];
    #endif

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

//LOAD_CAMERA
#ifdef CAMERA_QUEUE_POLARIZED

#define LOAD_CAMERA(RAY, QUEUE, IDX) \
CameraRay RAY = createCameraRay(\
    vec3(QUEUE.posX[IDX], QUEUE.posY[IDX], QUEUE.posZ[IDX]),\
    vec3(QUEUE.dirX[IDX], QUEUE.dirY[IDX], QUEUE.dirZ[IDX]),\
    vec3(QUEUE.polX[IDX], QUEUE.polY[IDX], QUEUE.polZ[IDX]),\
    mat4(\
        QUEUE.m00[IDX], QUEUE.m10[IDX], QUEUE.m20[IDX], QUEUE.m30[IDX],\
        QUEUE.m01[IDX], QUEUE.m11[IDX], QUEUE.m21[IDX], QUEUE.m31[IDX],\
        QUEUE.m02[IDX], QUEUE.m12[IDX], QUEUE.m22[IDX], QUEUE.m32[IDX],\
        QUEUE.m03[IDX], QUEUE.m13[IDX], QUEUE.m23[IDX], QUEUE.m33[IDX]\
    ),\
    QUEUE.contrib[IDX], QUEUE.timeDelta[IDX],\
    vec3(QUEUE.hitPosX[IDX], QUEUE.hitPosY[IDX], QUEUE.hitPosZ[IDX]),\
    vec3(QUEUE.hitDirX[IDX], QUEUE.hitDirY[IDX], QUEUE.hitDirZ[IDX]),\
    vec3(QUEUE.hitNrmX[IDX], QUEUE.hitNrmY[IDX], QUEUE.hitNrmZ[IDX]),\
    vec3(QUEUE.hitPolX[IDX], QUEUE.hitPolY[IDX], QUEUE.hitPolZ[IDX]));

#else

#define LOAD_CAMERA(RAY, QUEUE, IDX) \
CameraRay RAY = createCameraRay(\
    vec3(QUEUE.posX[IDX], QUEUE.posY[IDX], QUEUE.posZ[IDX]),\
    vec3(QUEUE.dirX[IDX], QUEUE.dirY[IDX], QUEUE.dirZ[IDX]),\
    QUEUE.contrib[IDX], QUEUE.timeDelta[IDX],\
    vec3(QUEUE.hitPosX[IDX], QUEUE.hitPosY[IDX], QUEUE.hitPosZ[IDX]),\
    vec3(QUEUE.hitDirX[IDX], QUEUE.hitDirY[IDX], QUEUE.hitDirZ[IDX]),\
    vec3(QUEUE.hitNrmX[IDX], QUEUE.hitNrmY[IDX], QUEUE.hitNrmZ[IDX]));

#endif

//SAVE_CAMERA
#if defined(POLARIZATION) && defined(CAMERA_QUEUE_POLARIZED)

#define SAVE_CAMERA(RAY, QUEUE, IDX)\
QUEUE.posX[IDX] = RAY.position.x;\
QUEUE.posY[IDX] = RAY.position.y;\
QUEUE.posZ[IDX] = RAY.position.z;\
QUEUE.dirX[IDX] = RAY.direction.x;\
QUEUE.dirY[IDX] = RAY.direction.y;\
QUEUE.dirZ[IDX] = RAY.direction.z;\
QUEUE.contrib[IDX] = RAY.contrib;\
QUEUE.timeDelta[IDX] = RAY.timeDelta;\
QUEUE.polX[IDX] = RAY.polRef.x;\
QUEUE.polY[IDX] = RAY.polRef.y;\
QUEUE.polZ[IDX] = RAY.polRef.z;\
QUEUE.m00[IDX] = RAY.mueller[0][0];\
QUEUE.m01[IDX] = RAY.mueller[1][0];\
QUEUE.m02[IDX] = RAY.mueller[2][0];\
QUEUE.m03[IDX] = RAY.mueller[3][0];\
QUEUE.m10[IDX] = RAY.mueller[0][1];\
QUEUE.m11[IDX] = RAY.mueller[1][1];\
QUEUE.m12[IDX] = RAY.mueller[2][1];\
QUEUE.m13[IDX] = RAY.mueller[3][1];\
QUEUE.m20[IDX] = RAY.mueller[0][2];\
QUEUE.m21[IDX] = RAY.mueller[1][2];\
QUEUE.m22[IDX] = RAY.mueller[2][2];\
QUEUE.m23[IDX] = RAY.mueller[3][2];\
QUEUE.m30[IDX] = RAY.mueller[0][3];\
QUEUE.m31[IDX] = RAY.mueller[1][3];\
QUEUE.m32[IDX] = RAY.mueller[2][3];\
QUEUE.m33[IDX] = RAY.mueller[3][3];\
QUEUE.hitPolX[IDX] = RAY.hit.polRef.x;\
QUEUE.hitPolY[IDX] = RAY.hit.polRef.y;\
QUEUE.hitPolZ[IDX] = RAY.hit.polRef.z;\
QUEUE.hitPosX[IDX] = RAY.hit.position.x;\
QUEUE.hitPosY[IDX] = RAY.hit.position.y;\
QUEUE.hitPosZ[IDX] = RAY.hit.position.z;\
QUEUE.hitDirX[IDX] = RAY.hit.direction.x;\
QUEUE.hitDirY[IDX] = RAY.hit.direction.y;\
QUEUE.hitDirZ[IDX] = RAY.hit.direction.z;\
QUEUE.hitNrmX[IDX] = RAY.hit.normal.x;\
QUEUE.hitNrmY[IDX] = RAY.hit.normal.y;\
QUEUE.hitNrmZ[IDX] = RAY.hit.normal.z;

#elif defined(LIGHT_QUEUE_POLARIZED)

#error "Unexpected mismatch between POLARIZATION and CAMERA_QUEUE_POLARIZED"

#else

#define SAVE_CAMERA(RAY, QUEUE, IDX)\
QUEUE.posX[IDX] = RAY.position.x;\
QUEUE.posY[IDX] = RAY.position.y;\
QUEUE.posZ[IDX] = RAY.position.z;\
QUEUE.dirX[IDX] = RAY.direction.x;\
QUEUE.dirY[IDX] = RAY.direction.y;\
QUEUE.dirZ[IDX] = RAY.direction.z;\
QUEUE.contrib[IDX] = RAY.contrib;\
QUEUE.timeDelta[IDX] = RAY.timeDelta;\
QUEUE.hitPosX[IDX] = RAY.hit.position.x;\
QUEUE.hitPosY[IDX] = RAY.hit.position.y;\
QUEUE.hitPosZ[IDX] = RAY.hit.position.z;\
QUEUE.hitDirX[IDX] = RAY.hit.direction.x;\
QUEUE.hitDirY[IDX] = RAY.hit.direction.y;\
QUEUE.hitDirZ[IDX] = RAY.hit.direction.z;\
QUEUE.hitNrmX[IDX] = RAY.hit.normal.x;\
QUEUE.hitNrmY[IDX] = RAY.hit.normal.y;\
QUEUE.hitNrmZ[IDX] = RAY.hit.normal.z;

#endif

#endif
