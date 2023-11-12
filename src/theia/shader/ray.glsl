#ifndef _RAY_INCLUDE
#define _RAY_INCLUDE

#include "material.glsl"

#ifndef N_PHOTONS
#define N_PHOTONS 4
#endif

struct Photon {
    float wavelength;           // 4 bytes
    //start time + travel time
    float time;                 // 4 bytes

    //to (hopefully) be more numerically stable and a bit faster on the
    //calculations, split lin and exp factors. Total contribrution:
    // L(path)/p(path) = lin_contrib * exp(log_contrib)
    float lin_contribution;     // 4 bytes
    float log_contribution;     // 4 bytes
    
    //cache medium constants to ease memory pressure
    MediumConstants constants;  //16 bytes
};                      // TOTAL: 32 bytes

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

struct PhotonHit {
    float wavelength;   // 4 bytes
    float time;         // 4 bytes
    // L(path)/p(path)
    float contribution; // 4 bytes
};              // TOTAL: 12 bytes

PhotonHit createHit(Photon photon) {
    float contribution = exp(photon.log_contribution) * photon.lin_contribution;
    return PhotonHit(
        photon.wavelength,
        photon.time,
        contribution
    );
}

struct RayHit {
    //in object space (no trafo)
    vec3 position;              //12 bytes
    vec3 direction;             //12 bytes
    vec3 normal;                //12 bytes

    PhotonHit hits[N_PHOTONS];  //N * 12 bytes
};                      // TOTAL: 36 + N*12 bytes (84 bytes)

#endif
