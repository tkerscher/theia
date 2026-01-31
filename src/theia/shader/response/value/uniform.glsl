#ifndef _INCLUDE_RESPONSE_UNIFORM
#define _INCLUDE_RESPONSE_UNIFORM

float responseValue(HitItem hit, uint idx, inout uint dim) {
    return hit.contrib;
}

#endif
