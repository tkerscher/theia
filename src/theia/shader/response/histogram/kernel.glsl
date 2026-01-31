#ifndef _INCLUDE_RESPONSE_KERNEL_HISTOGRAM
#define _INCLUDE_RESPONSE_KERNEL_HISTOGRAM

#include "util/buffers.glsl"
#include "util/launchid.glsl"

uniform ResponseParams {
    uvec2 bufferAdr;
    uint binCount;
    uint histCount;

    float t0;
    float binSize;

    float kernelBandwidth;
    float kernelSupport;
} responseParams;

void updateBin(uint hist, uint bin, float value) {
    BinBuffer bins = BinBuffer(responseParams.bufferAdr);
    uint i = hist * responseParams.binCount + bin;
    atomicAdd(bins.values[i], value);
}

//For now we will only support gaussian kernel
float kernelCdf(float x) {
    //numerical approximation for normal distribution CDF from:
    //H. Vazquez-Leal et al.: "High Accurate Simple Approximation of Normal
    //Distribution Integral" (2011); Eq. 4.3
    const float c1 = 7.779374467827938; // = 39 / 2sqrt(2pi)
    const float c2 = 55.5;
    const float c3 = 0.1257926109373887; // = 35 / 111sqrt(2pi)
    return 0.5 + 0.5 * tanh(c1 * x - c2 * atan(c3 * x));
}

void response(HitItem item, uint idx, inout uint dim) {
    //get response value
    float value = responseValue(item, idx, dim);
    //ignore zero and NaNs
    if (!(value != 0.0)) return;

    //hash launch id to select a random histogram for storing the value
    //this is to avoid serialization of atomicAdds of the same bin
    uint histIdx = getScrambledLaunchId(responseParams.histCount);

    //calculate which bins will be affected
    float t = item.time - responseParams.t0;
    int firstBin = int(floor((t - responseParams.kernelSupport) / responseParams.binSize));
    int lastBin = int(ceil((t + responseParams.kernelSupport) / responseParams.binSize));
    //clamp bins
    firstBin = max(firstBin, 0);
    lastBin = min(lastBin, int(responseParams.binCount)) - 1;

    //update histogram
    t = firstBin * responseParams.binSize + responseParams.t0 - item.time;
    float h = 1.0 / responseParams.kernelBandwidth;
    float prev_cdf = kernelCdf(t * h);
    for (int i = firstBin; i <= lastBin; ++i) {
        //calculate bin weight
        // w = CDF( t[i+1] ) - CDF( t[i] )
        t += responseParams.binSize;
        float cdf = kernelCdf(t * h);
        float w = cdf - prev_cdf;
        prev_cdf = cdf;
        //update bin
        updateBin(histIdx, i, w * value);
    }
}

#endif
