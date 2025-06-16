#ifndef _INCLUDE_RESPONSE_KERNEL_HISTOGRAM
#define _INCLUDE_RESPONSE_KERNEL_HISTOGRAM

struct Histogram {
    float bins[N_BINS];
};
layout(scalar) writeonly buffer HistogramOut {
    Histogram histsOut[];
};

layout(scalar) uniform ResponseParams {
    //histogram params
    float t0;
    float binSize;

    //kernel params
    float kernelBandwidth;
    float kernelSupport;
} responseParams;

#ifdef RESPONSE_KERNEL_HISTOGRAM_USE_SHARED_MEMORY

shared float localHist[N_BINS];

void initResponse() {
    //clear local hist
    [[unroll]] for (uint i = gl_LocalInvocationID.x; i < N_BINS; i += BLOCK_SIZE) {
        localHist[i] = 0.0;
    }

    //block until local hist is initialized
    memoryBarrierShared();
    barrier();
}

void updateBin(int bin, float value) {
    atomicAdd(localHist[bin], value);
}

void finalizeResponse() {
    //ensure we're finished with the local histogram
    memoryBarrierShared();
    barrier();

    //copy local histogram from shared memory to global memory
    uint histId = gl_WorkGroupID.x;
    [[unroll]] for (uint i = gl_LocalInvocationID.x; i < N_BINS; i += BLOCK_SIZE) {
        histsOut[histId].bins[i] = localHist[i];
    }
}

#else //ifdef RESPONSE_KERNEL_HISTOGRAM_USE_SHARED_MEMORY

void initResponse() {}

void updateBin(int bin, float value) {
    atomicAdd(histsOut[gl_WorkGroupID.x].bins[bin], value);
}

void finalizeResponse() {}

#endif

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

    //calculate which bins will be affected
    float t = item.time - responseParams.t0;
    int firstBin = int(floor((t - responseParams.kernelSupport) / responseParams.binSize));
    int lastBin = int(ceil((t + responseParams.kernelSupport) / responseParams.binSize));
    //clamp bins
    firstBin = max(firstBin, 0);
    lastBin = min(lastBin, N_BINS) - 1;

    //update histogram
    t = firstBin * responseParams.binSize + responseParams.t0 - item.time;
    // t = firstBin * responseParams.binSize - t;
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
        updateBin(i, w * value);
    }
}

#endif
