#ifndef _RAY_INCLUDE
#define _RAY_INCLUDE

#include "material.glsl"

//Check macro configs are set
#ifndef N_PHOTONS
#error "missing macro setting: N_PHOTONS"
#endif
#ifndef QUEUE_SIZE
#error "missing macro settings: QUEUE_SIZE"
#endif

struct Photon {
    float wavelength;           // 4 bytes
    //start time + travel time
    float time;                 // 4 bytes

    //to (hopefully) be more numerically stable and a bit faster on the
    //calculations, split lin and exp factors. Total contribrution:
    // L(path)/p(path) = lin_contrib * exp(log_contrib)
    float lin_contrib;          // 4 bytes
    float log_contrib;          // 4 bytes
    
    //cache medium constants to ease memory pressure
    MediumConstants constants;  //16 bytes
};                      // TOTAL: 32 bytes

struct Ray {
    vec3 position;              //12 bytes
    vec3 direction;             //12 bytes
    uint rngStream;             // 4 bytes
    uint rngCount;              // 4 bytes

    //A plane buffer reference would have an alignment of 8 bytes (uint64).
    //Moreover, Ray would also get an alignment of 8 bytes. To make things more
    //simple, we'll use uvec2 instead and thus have an alignment of only 4 bytes
    uvec2 medium;               // 8 bytes
    Photon photons[N_PHOTONS];  // N * 32 bytes
};          // TOTAL: 40 + N*32 bytes (168 bytes)

struct RayQueue {
    //ray (40 bytes)
    float posX[QUEUE_SIZE];
    float posY[QUEUE_SIZE];
    float posZ[QUEUE_SIZE];
    float dirX[QUEUE_SIZE];
    float dirY[QUEUE_SIZE];
    float dirZ[QUEUE_SIZE];
    uint rngStream[QUEUE_SIZE];
    uint rngCount[QUEUE_SIZE];
    uint mediumLo[QUEUE_SIZE];
    uint mediumHi[QUEUE_SIZE];
    //photon (16*N_PHOTONS bytes)
    float wavelength[N_PHOTONS][QUEUE_SIZE];
    float time[N_PHOTONS][QUEUE_SIZE];
    float lin_contrib[N_PHOTONS][QUEUE_SIZE];
    float log_contrib[N_PHOTONS][QUEUE_SIZE];
    //medium constants (16*N_PHOTONS bytes)
    float n[N_PHOTONS][QUEUE_SIZE];
    float vg[N_PHOTONS][QUEUE_SIZE];
    float mu_s[N_PHOTONS][QUEUE_SIZE];
    float mu_e[N_PHOTONS][QUEUE_SIZE];
};  //TOTAL: 40 + 32*N_PHOTONS bytes

Photon createPhoton(
    Medium medium,
    float wavelength,
    float startTime,
    float lin_contrib,
    float log_contrib
) {
    return Photon(
        wavelength,
        startTime,
        lin_contrib,
        log_contrib,
        lookUpMedium(medium, wavelength)
    );
}

//Thanks to GLSL not having references, but only passes variables by copy we
//would have to bet on the compiler being smart enough to inline the function
//and not copy around several MB of data, when we only want to read a few bytes.
//Since I don't like to bet, here are some ugly macros instead

#define LOAD_RAY(RAY, QUEUE, IDX) \
Photon RAY##_photons[N_PHOTONS];\
for (uint i = 0; i < N_PHOTONS; ++i) {\
    RAY##_photons[i] = Photon(\
        QUEUE.wavelength[i][IDX],\
        QUEUE.time[i][IDX],\
        QUEUE.lin_contrib[i][IDX],\
        QUEUE.log_contrib[i][IDX],\
        MediumConstants(\
            QUEUE.n[i][IDX],\
            QUEUE.vg[i][IDX],\
            QUEUE.mu_s[i][IDX],\
            QUEUE.mu_e[i][IDX]));\
}\
Ray RAY = Ray(\
    vec3(QUEUE.posX[IDX], QUEUE.posY[IDX], QUEUE.posZ[IDX]),\
    vec3(QUEUE.dirX[IDX], QUEUE.dirY[IDX], QUEUE.dirZ[IDX]),\
    QUEUE.rngStream[IDX], QUEUE.rngCount[IDX],\
    uvec2(QUEUE.mediumLo[IDX], QUEUE.mediumHi[IDX]),\
    RAY##_photons);

#define SAVE_RAY(RAY, QUEUE, IDX)\
for (uint i = 0; i < N_PHOTONS; ++i) {\
    QUEUE.wavelength[i][IDX] = RAY.photons[i].wavelength;\
    QUEUE.time[i][IDX] = RAY.photons[i].time;\
    QUEUE.lin_contrib[i][IDX] = RAY.photons[i].lin_contrib;\
    QUEUE.log_contrib[i][IDX] = RAY.photons[i].log_contrib;\
    QUEUE.n[i][IDX] = RAY.photons[i].constants.n;\
    QUEUE.vg[i][IDX] = RAY.photons[i].constants.vg;\
    QUEUE.mu_s[i][IDX] = RAY.photons[i].constants.mu_s;\
    QUEUE.mu_e[i][IDX] = RAY.photons[i].constants.mu_e;\
}\
QUEUE.posX[IDX] = RAY.position.x;\
QUEUE.posY[IDX] = RAY.position.y;\
QUEUE.posZ[IDX] = RAY.position.z;\
QUEUE.dirX[IDX] = RAY.direction.x;\
QUEUE.dirY[IDX] = RAY.direction.y;\
QUEUE.dirZ[IDX] = RAY.direction.z;\
QUEUE.rngStream[IDX] = RAY.rngStream;\
QUEUE.rngCount[IDX] = RAY.rngCount;\
QUEUE.mediumLo[IDX] = RAY.medium.x;\
QUEUE.mediumHi[IDX] = RAY.medium.y;

#endif
