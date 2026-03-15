#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV_CASCADE_COMMON
#define _INCLUDE_LIGHTSOURCE_CHERENKOV_CASCADE_COMMON

uniform CascadeParams {
    //geometric properties
    vec3 startPosition;
    float startTime;
    vec3 direction;

    //light yield parameters
    float effectiveLength;
    //angular emission profile
    float a_angular;
    float b_angular;
    //longitudinal profile (Eq. 4.10 in [1])
    float a_long; //gamma dist shape
    float b_long; //gamma dist scale length [m]

    //For better compatability, we use the same defintion of b_long as ice tray.
    //That however means that we combined two things here:
    // - The actual b parameter of the gamma distribution (Eq. 4.10 in [1])
    // - Multiplying the radiation length X_0 to convert cascade depth to actual depth
    //For gamma distributions the following holds:
    // gamma(a, b) = gamma(a, 1) / b
    //If we also multiply X_0, we get ice tray's version of b (called b' in the
    //next line):
    // X_0 * gamma(a, b) = X_0 * gamma(a, 1) / b = b' * gamma(a, 1)

    uint mediumIdx;
} cascade;

#endif
