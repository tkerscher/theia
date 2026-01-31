#ifndef _INCLUDE_RESPONSE_INTEGRATE
#define _INCLUDE_RESPONSE_INTEGRATE

#include "util/buffers.glsl"
#include "util/launchid.glsl"

uniform ResponseParams {
    uvec2 bufferAdr;
    uint bufferCount;
    uint binCount;
} responseParams;

void response(HitItem item, uint idx, inout uint dim) {
    //process hit
    float value = responseValue(item, idx, dim);
    //ignore zero and NaNs
    if (!(value != 0.0)) return;

    //hash launch id to select a random histogram for storing the value
    //this is to avoid serialization of atomicAdds of the same bin
    uint bufferIdx = getScrambledLaunchId(responseParams.bufferCount);

    //store value
    BinBuffer bins = BinBuffer(responseParams.bufferAdr);
    #ifdef RESPONSE_INTEGRATE_ALL
    atomicAdd(bins.values[bufferIdx], value);
    #else
    uint i = bufferIdx * responseParams.binCount + item.objectId;
    atomicAdd(bins.values[i], value);
    #endif
}

#endif
