#ifndef _INCLUDE_RESPONSE_HISTOGRAM
#define _INCLUDE_RESPONSE_HISTOGRAM

#include "util/buffers.glsl"
#include "util/launchid.glsl"

uniform ResponseParams {
    uvec2 bufferAdr;
    uint binCount;
    uint histCount;

    float t0;
    float binSize;
} responseParams;

void updateBin(uint hist, uint bin, float value) {
    BinBuffer bins = BinBuffer(responseParams.bufferAdr);
    uint i = hist * responseParams.binCount + bin;
    atomicAdd(bins.values[i], value);
}

void response(HitItem item, uint idx, inout uint dim) {
    //get response value
    float value = responseValue(item, idx, dim);
    //ignore zero and NaNs
    if (!(value != 0.0)) return;

    //hash launch id to select a random histogram for storing the value
    //this is to avoid serialization of atomicAdds of the same bin
    uint histIdx = getScrambledLaunchId(responseParams.histCount);

    //update histogram
    uint bin = int(floor((item.time - responseParams.t0) / responseParams.binSize));
    if (bin >= 0 && bin < responseParams.binCount) {
        updateBin(histIdx, bin, value);
    }
}

#endif
