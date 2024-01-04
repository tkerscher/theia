#ifndef _INCLUDE_RESPONSE_LAMBERT
#define _INCLUDE_RESPONSE_LAMBERT

float responseValue(HitItem hit) {
    return -dot(hit.direction, hit.normal) * hit.contrib;
}

#endif
