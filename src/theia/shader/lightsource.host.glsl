#ifndef _INCLUDE_LIGHTSOURCE_HOST
#define _INCLUDE_LIGHTSOURCE_HOST

//read and return rays from a buffer

layout(scalar) readonly buffer SourceRayQueue {
    //ray
    float posX[QUEUE_SIZE];
    float posY[QUEUE_SIZE];
    float posZ[QUEUE_SIZE];
    float dirX[QUEUE_SIZE];
    float dirY[QUEUE_SIZE];
    float dirZ[QUEUE_SIZE];
    //photon
    float wavelength[N_PHOTONS][QUEUE_SIZE];
    float time[N_PHOTONS][QUEUE_SIZE];
    float lin_contrib[N_PHOTONS][QUEUE_SIZE];
    float log_contrib[N_PHOTONS][QUEUE_SIZE];
} sourceQueue;

SourceRay sampleLight() {
    uint idx = gl_GlobalInvocationID.x;
    //load photons
    SourcePhoton photons[N_PHOTONS];
    for (uint i = 0; i < N_PHOTONS; ++i) {
        photons[i] = SourcePhoton(
            sourceQueue.wavelength[i][idx],
            sourceQueue.time[i][idx],
            sourceQueue.lin_contrib[i][idx],
            sourceQueue.log_contrib[i][idx]
        );
    }
    //load rays
    SourceRay ray = SourceRay(
        vec3(sourceQueue.posX[idx], sourceQueue.posY[idx], sourceQueue.posZ[idx]),
        vec3(sourceQueue.dirX[idx], sourceQueue.dirY[idx], sourceQueue.dirZ[idx]),
        photons
    );
    //return ray
    return ray;
}

#endif
