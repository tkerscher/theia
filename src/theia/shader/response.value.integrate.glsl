#ifndef _INCLUDE_RESPONSE_INTEGRATE
#define _INCLUDE_RESPONSE_INTEGRATE

writeonly buffer ValueOut {
    float values[];
} valueOut;

void initResponse() {}

void response(HitItem hit, uint idx, inout uint dim) {
    float value = responseValue(hit, idx, dim);
    #ifdef RESPONSE_INTEGRATE_ALL
    atomicAdd(valueOut.values[0], value);
    #else
    atomicAdd(valueOut.values[hit.objectId], value);
    #endif
}

void finalizeResponse() {}

#endif
