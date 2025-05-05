#ifndef _INCLUDE_RESPONSE_QUEUE
#define _INCLUDE_RESPONSE_QUEUE

#include "response.common.glsl"

struct HitQueue {
    //ray
    float posX[HIT_QUEUE_SIZE];
    float posY[HIT_QUEUE_SIZE];
    float posZ[HIT_QUEUE_SIZE];
    float dirX[HIT_QUEUE_SIZE];
    float dirY[HIT_QUEUE_SIZE];
    float dirZ[HIT_QUEUE_SIZE];
    float nrmX[HIT_QUEUE_SIZE];
    float nrmY[HIT_QUEUE_SIZE];
    float nrmZ[HIT_QUEUE_SIZE];
    //polarization
#ifdef HIT_QUEUE_POLARIZED
    float stokesI[HIT_QUEUE_SIZE];
    float stokesQ[HIT_QUEUE_SIZE];
    float stokesU[HIT_QUEUE_SIZE];
    float stokesV[HIT_QUEUE_SIZE];
    float polX[HIT_QUEUE_SIZE];
    float polY[HIT_QUEUE_SIZE];
    float polZ[HIT_QUEUE_SIZE];
#endif
    //photon
    float wavelength[HIT_QUEUE_SIZE];
    float time[HIT_QUEUE_SIZE];
    float contrib[HIT_QUEUE_SIZE];

    int objectId[HIT_QUEUE_SIZE];
};

//Thanks to GLSL not having references, but only passes variables by copy we
//would have to bet on the compiler being smart enough to inline the function
//and not copy around several MB of data, when we only want to read a few bytes.
//Since I don't like to bet, here are some ugly macros instead

//LOAD_HIT()
#if defined(HIT_QUEUE_POLARIZED) && defined(POLARIZATION)

#define LOAD_HIT(HIT, QUEUE, IDX) \
HitItem HIT = HitItem(\
    vec3(QUEUE.posX[IDX], QUEUE.posY[IDX], QUEUE.posZ[IDX]),\
    vec3(QUEUE.dirX[IDX], QUEUE.dirY[IDX], QUEUE.dirZ[IDX]),\
    vec3(QUEUE.nrmX[IDX], QUEUE.nrmY[IDX], QUEUE.nrmZ[IDX]),\
    vec4(QUEUE.stokesI[IDX], QUEUE.stokesQ[IDX], QUEUE.stokesU[IDX], QUEUE.stokesV[IDX]),\
    vec3(QUEUE.polX[IDX], QUEUE.polY[IDX], QUEUE.polZ[IDX]),\
    QUEUE.wavelength[IDX], QUEUE.time[IDX], QUEUE.contrib[IDX], QUEUE.objectId[IDX]);

#elif defined(POLARIZATION)

#error "unexpected polarization configuration"

#else

#define LOAD_HIT(HIT, QUEUE, IDX) \
HitItem HIT = HitItem(\
    vec3(QUEUE.posX[IDX], QUEUE.posY[IDX], QUEUE.posZ[IDX]),\
    vec3(QUEUE.dirX[IDX], QUEUE.dirY[IDX], QUEUE.dirZ[IDX]),\
    vec3(QUEUE.nrmX[IDX], QUEUE.nrmY[IDX], QUEUE.nrmZ[IDX]),\
    QUEUE.wavelength[IDX], QUEUE.time[IDX], QUEUE.contrib[IDX], QUEUE.objectId[IDX]);

#endif

//SAVE_HIT()
#if defined(HIT_QUEUE_POLARIZED) && defined(POLARIZATION)

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
QUEUE.stokesI[IDX] = HIT.stokes.x;\
QUEUE.stokesQ[IDX] = HIT.stokes.y;\
QUEUE.stokesU[IDX] = HIT.stokes.z;\
QUEUE.stokesV[IDX] = HIT.stokes.w;\
QUEUE.polX[IDX] = HIT.polRef.x;\
QUEUE.polY[IDX] = HIT.polRef.y;\
QUEUE.polZ[IDX] = HIT.polRef.z;\
QUEUE.wavelength[IDX] = HIT.wavelength;\
QUEUE.time[IDX] = HIT.time;\
QUEUE.contrib[IDX] = HIT.contrib;\
QUEUE.objectId[IDX] = HIT.objectId;

#elif defined(HIT_QUEUE_POLARIZED)

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
QUEUE.stokesI[IDX] = 1.0;\
QUEUE.stokesQ[IDX] = 0.0;\
QUEUE.stokesU[IDX] = 0.0;\
QUEUE.stokesV[IDX] = 0.0;\
QUEUE.polX[IDX] = 0.0;\
QUEUE.polY[IDX] = 0.0;\
QUEUE.polZ[IDX] = 0.0;\
QUEUE.wavelength[IDX] = HIT.wavelength;\
QUEUE.time[IDX] = HIT.time;\
QUEUE.contrib[IDX] = HIT.contrib;\
QUEUE.objectId[IDX] = HIT.objectId;

#else

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
QUEUE.contrib[IDX] = HIT.contrib;\
QUEUE.objectId[IDX] = HIT.objectId;

#endif

#endif
