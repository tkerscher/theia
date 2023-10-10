#ifndef _RAY_INCLUDE
#define _RAY_INCLUDE

#include "material.glsl"

#ifndef N_PHOTONS
#define N_PHOTONS 4
#endif

struct Photon {
    float wavelength;           // 4 bytes
    float travelTime;           // 4 bytes
    //keeping radiance and throughput separate allows to MIS light (if we want to...)
    float log_radiance;         // 4 bytes

    //total throughput: linT * exp(logT)
    //  hope this makes the estimate more precise...
    float T_lin;                // 4 bytes
    float T_log;                // 4 bytes
    
    MediumConstants constants;  //16 bytes
};                      // TOTAL: 36 bytes

Photon createPhoton(
    Medium medium,
    float wavelength,
    float startTime,
    float log_radiance,
    float prob
) {
    return Photon(
        wavelength,
        startTime,
        log_radiance,
        prob, // throughput
        0.0,  // log throughput
        lookUpMedium(medium, wavelength)
    );
}

struct Ray{
    vec3 position;              //12 bytes
    vec3 direction;             //12 bytes
    uint rngIdx;                // 4 bytes

    //A plane buffer reference would have an alignment of 8 bytes (uint64).
    //Moreover, Ray would also get an alignment of 8 bytes. To make things more
    //simple, we'll use uvec2 instead and thus have an alignment of only 4 bytes
    uvec2 medium;               // 8 bytes
    Photon photons[N_PHOTONS];  // N * 36 bytes
};          // TOTAL: 36 + N*36 bytes (180 bytes)

#endif
