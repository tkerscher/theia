#ifndef _INCLUDE_RESPONSE_UNIFORM
#define _INCLUDE_RESPONSE_UNIFORM

float responseValue(HitItem hit, uint idx, inout uint dim) {
    #ifdef RAY_PARTICLE
    return 1.0;
    #else
    return hit.contrib;
    #endif
}

#endif
