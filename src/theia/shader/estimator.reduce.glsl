#extension GL_EXT_scalar_block_layout : require

//default settings; overwritten by preamble
#ifndef LOCAL_SIZE
#define LOCAL_SIZE 32
#endif
#ifndef N_BINS
#define N_BINS 256
#endif

layout(local_size_x = LOCAL_SIZE) in;

struct Histogram {
    float bin[N_BINS];
};

layout(scalar) readonly buffer HistogramIn {
    Histogram histIn[];
};
layout(scalar) writeonly buffer HistogramOut {
    Histogram histOut;
};

layout(scalar) uniform Params {
    float norm;
    uint nHist;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= N_BINS)
        return;
    
    //add same bin across all inputs and save in output
    float result = 0.0;
    for (uint i = 0; i < nHist; ++i) {
        result += histIn[i].bin[idx] * norm;
    }
    histOut.bin[idx] += result;
}
