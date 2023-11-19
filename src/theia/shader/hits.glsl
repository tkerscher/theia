#ifndef _HITS_INCLUDE
#define _HITS_INCLUDE

//check macro settings are set
#ifndef N_PHOTONS
#error "missing macro setting: N_PHOTONS"
#endif
#ifndef HIT_QUEUE_SIZE
#error "missing macro settings: QUEUE_SIZE"
#endif

struct PhotonHit {
    float wavelength;   // 4 bytes
    float time;         // 4 bytes
    // L(path)/p(path)
    float contribution; // 4 bytes
};              // TOTAL: 12 bytes

struct RayHit {
    //in object space (no trafo)
    vec3 position;              //12 bytes
    vec3 direction;             //12 bytes
    vec3 normal;                //12 bytes

    PhotonHit hits[N_PHOTONS];  //N * 12 bytes
};                      // TOTAL: 36 + N*12 bytes (84 bytes)

struct HitQueue {
    //ray hit (36 bytes)
    float posX[HIT_QUEUE_SIZE];
    float posY[HIT_QUEUE_SIZE];
    float posZ[HIT_QUEUE_SIZE];
    float dirX[HIT_QUEUE_SIZE];
    float dirY[HIT_QUEUE_SIZE];
    float dirZ[HIT_QUEUE_SIZE];
    float nrmX[HIT_QUEUE_SIZE];
    float nrmY[HIT_QUEUE_SIZE];
    float nrmZ[HIT_QUEUE_SIZE];
    //photon hits (12*N_PHOTONS*N_HITS bytes)
    float wavelength[N_PHOTONS][HIT_QUEUE_SIZE];
    float time[N_PHOTONS][HIT_QUEUE_SIZE];
    float contribution[N_PHOTONS][HIT_QUEUE_SIZE];
};  //TOTAL 36 + 12*N_PHOTONS bytes

#define SAVE_HIT(POS, DIR, NRM, HITS, QUEUE, IDX) \
QUEUE.posX[IDX] = POS.x;\
QUEUE.posY[IDX] = POS.y;\
QUEUE.posZ[IDX] = POS.z;\
QUEUE.dirX[IDX] = DIR.x;\
QUEUE.dirY[IDX] = DIR.y;\
QUEUE.dirZ[IDX] = DIR.z;\
QUEUE.nrmX[IDX] = NRM.x;\
QUEUE.nrmY[IDX] = NRM.y;\
QUEUE.nrmZ[IDX] = NRM.z;\
for (uint i = 0; i < N_PHOTONS; ++i) {\
    QUEUE.wavelength[i][IDX] = HITS[i].wavelength;\
    QUEUE.time[i][IDX] = HITS[i].time;\
    QUEUE.contribution[i][IDX] = HITS[i].contribution;\
}

#endif
