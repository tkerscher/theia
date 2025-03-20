//check macro settings
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE is not defined"
#endif
#ifndef N_BINS
#error "N_BINS is not defined"
#endif
#ifndef VALUE_QUEUE_SIZE
#error "VALUE_QUEUE_SIZE is not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

//local histogram
shared float localHist[N_BINS];
//global histograms
struct Histogram {
    float bins[N_BINS];
};
layout(scalar) writeonly buffer HistogramOut {
    Histogram histsOut[];
};

layout(scalar) readonly buffer ValueIn {
    uint valueCount;
    float value[VALUE_QUEUE_SIZE];
    float time[VALUE_QUEUE_SIZE];
};

layout(scalar) uniform Parameters {
    float t0;
    float binSize;
};

void main() {
    uint i_item = gl_GlobalInvocationID.x;
    uint i_local = gl_LocalInvocationID.x;

    //clear local hist
    [[unroll]] for (uint i = i_local; i < N_BINS; i += BLOCK_SIZE) {
        localHist[i] = 0.0;
    }
    memoryBarrierShared();
    barrier();

    //we can't use early returns in combination with barrier()
    //-> if guard
    if (i_item < valueCount) {
        //calculate affected bin
        uint bin = int(floor((time[i_item] - t0) / binSize));
        //update histogram if bin is in range
        if (bin >= 0 && bin < N_BINS) {
            atomicAdd(localHist[bin], value[i_item]);
        }
    }

    //ensure we're finished with the local histogram
    memoryBarrierShared();
    barrier();

    //copy local histogram from shared memory to global memory
    uint histId = gl_WorkGroupID.x;
    [[unroll]] for (uint i = i_local; i < N_BINS; i += BLOCK_SIZE) {
        histsOut[histId].bins[i] = localHist[i];
    }
}
