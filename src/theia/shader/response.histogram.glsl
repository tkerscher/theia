#ifndef _INCLUDE_RESPONSE_HISTOGRAM
#define _INCLUDE_RESPONSE_HISTOGRAM

struct Histogram {
    float bins[N_BINS];
};
layout(scalar) writeonly buffer HistogramOut {
    Histogram histsOut[];   
};

layout(scalar) uniform ResponseParams {
    float t0;
    float binSize;
} responseParams;

#ifdef RESPONSE_HISTOGRAM_USE_SHARED_MEMORY

shared float localHist[N_BINS];

void initResponse() {
    //clear local hist
    [[unroll]] for (uint i = gl_LocalInvocationID.x; i < N_BINS; i += BLOCK_SIZE) {
        localHist[i] = 0.0;
    }

    //halt tracer until local hist is initialized
    memoryBarrierShared();
    barrier();
}

void updateBin(uint bin, float value) {
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

#else //ifdef RESPONSE_HISTOGRAM_USE_SHARED_MEMORY

void initResponse() {}

void updateBin(uint bin, float value) {
    atomicAdd(histsOut[gl_WorkGroupID.x].bins[bin], value);
}

void finalizeResponse() {}

#endif

void response(HitItem item, uint dim, inout uint idx) {
    //get response value
    float value = responseValue(item, dim, idx);

    //update local history
    uint bin = int(floor((item.time - responseParams.t0) / responseParams.binSize));
    if (bin >= 0 && bin < N_BINS) {
        updateBin(bin, value);
    }
}

#endif
