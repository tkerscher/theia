#ifndef _INCLUDE_TRACER_PHOTON_QUEUE
#define _INCLUDE_TRACER_PHOTON_QUEUE

#include "ray.glsl"

struct PhotonQueue {
    float posX[PHOTON_QUEUE_SIZE];
    float posY[PHOTON_QUEUE_SIZE];
    float posZ[PHOTON_QUEUE_SIZE];
    float dirX[PHOTON_QUEUE_SIZE];
    float dirY[PHOTON_QUEUE_SIZE];
    float dirZ[PHOTON_QUEUE_SIZE];

    #ifndef USE_GLOBAL_MEDIUM
    uint mediumX[PHOTON_QUEUE_SIZE];
    uint mediumY[PHOTON_QUEUE_SIZE];
    #endif

    #ifdef POLARIZATION
    float stokesQ[PHOTON_QUEUE_SIZE];
    float stokesU[PHOTON_QUEUE_SIZE];
    float stokesV[PHOTON_QUEUE_SIZE];
    float polRefX[PHOTON_QUEUE_SIZE];
    float polRefY[PHOTON_QUEUE_SIZE];
    float polRefZ[PHOTON_QUEUE_SIZE];
    #endif

    float wavelength[PHOTON_QUEUE_SIZE];
    float time[PHOTON_QUEUE_SIZE];
    uint idx[PHOTON_QUEUE_SIZE]; //RNG
};

layout(scalar) readonly buffer PhotonQueueIn {
    uvec3 groupCount; //used for indirect dispatch

    uint count;
    PhotonQueue queue;
} photonQueueIn;
layout(scalar) buffer PhotonQueueOut {
    uvec3 groupCount; //used for indirect dispatch

    uint count;
    PhotonQueue queue;
} photonQueueOut;

void saveRay(const ForwardRay ray, uint idx) {
    uint i = atomicAdd(photonQueueOut.count, 1);
    //update group count
    if (i % BLOCK_SIZE == 0) {
        //ceil division of (i + 1) / BLOCK_SIZE; // atomicAdd() returns old value
        uint groups = i / BLOCK_SIZE + 1;
        atomicMax(photonQueueOut.groupCount.x, max(groups, 1));
    }

    photonQueueOut.queue.posX[i] = ray.state.position.x;
    photonQueueOut.queue.posY[i] = ray.state.position.y;
    photonQueueOut.queue.posZ[i] = ray.state.position.z;
    photonQueueOut.queue.dirX[i] = ray.state.direction.x;
    photonQueueOut.queue.dirY[i] = ray.state.direction.y;
    photonQueueOut.queue.dirZ[i] = ray.state.direction.z;

    #ifndef USE_GLOBAL_MEDIUM
    photonQueueOut.queue.mediumX[i] = ray.state.medium.x;
    photonQueueOut.queue.mediumY[i] = ray.state.medium.y;
    #endif

    #ifdef POLARIZATION
    //we assume stokes is already re-normalized
    photonQueueOut.queue.stokesQ[i] = ray.stokes.y;
    photonQueueOut.queue.stokesU[i] = ray.stokes.z;
    photonQueueOut.queue.stokesV[i] = ray.stokes.w;
    photonQueueOut.queue.polRefX[i] = ray.polRef.x;
    photonQueueOut.queue.polRefY[i] = ray.polRef.y;
    photonQueueOut.queue.polRefZ[i] = ray.polRef.z;
    #endif

    photonQueueOut.queue.wavelength[i] = ray.state.wavelength;
    photonQueueOut.queue.time[i] = ray.state.time;
    photonQueueOut.queue.idx[i] = idx;
}

ForwardRay loadRay(
    #ifdef USE_GLOBAL_MEDIUM
    Medium medium,
    #endif
    uint idx
) {
    vec3 position = vec3(
        photonQueueIn.queue.posX[idx],
        photonQueueIn.queue.posY[idx],
        photonQueueIn.queue.posZ[idx]
    );
    vec3 direction = vec3(
        photonQueueIn.queue.dirX[idx],
        photonQueueIn.queue.dirY[idx],
        photonQueueIn.queue.dirZ[idx]
    );

    float wavelength = photonQueueIn.queue.wavelength[idx];
    float time = photonQueueIn.queue.time[idx];

    #ifndef USE_GLOBAL_MEDIUM
    Medium medium = Medium(uvec2(
        photonQueueIn.queue.mediumX[idx],
        photonQueueIn.queue.mediumY[idx]
    ));
    #endif

    #ifdef POLARIZATION
    vec4 stokes = vec4(
        1.0,
        photonQueueIn.queue.stokesQ[idx],
        photonQueueIn.queue.stokesU[idx],
        photonQueueIn.queue.stokesV[idx]
    );
    vec3 polRef = vec3(
        photonQueueIn.queue.polRefX[idx],
        photonQueueIn.queue.polRefY[idx],
        photonQueueIn.queue.polRefZ[idx]
    );
    #endif

    SourceRay source = createSourceRay(
        position,
        direction,
        #ifdef POLARIZATION
        stokes,
        polRef,
        #endif
        time,
        1.0
    );
    MediumConstants consts = lookUpMedium(medium, wavelength);
    return createRay(source, medium, consts, wavelength);
}

#endif
