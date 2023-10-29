#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require

//default settings; overwritten by preamble
#ifndef LOCAL_SIZE
#define LOCAL_SIZE 32
#endif
#ifndef N_BINS
#define N_BINS 256
#endif

layout(local_size_x = LOCAL_SIZE) in;

layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer HistogramIn {
    float bin[N_BINS];
};
layout(buffer_reference, scalar, buffer_reference_align=4) writeonly buffer HistogramOut {
    float bin[N_BINS];
};

layout(push_constant, scalar) uniform Push {
    uvec2 histIn;           // 8 bytes
    uvec2 histOut;          // 8 bytes
    float normalization;    // 4 bytes
    uint nHist;             // 4 bytes
} params;           // TOTAL: 24 bytes

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= N_BINS)
        return;
    
    //fetch histograms
    HistogramIn histIn = HistogramIn(params.histIn);
    HistogramOut histOut = HistogramOut(params.histOut);

    //add same bin across all inputs and save in output
    float result = 0.0;
    for (uint i = 0; i < params.nHist; ++i) {
        result += histIn[i].bin[idx] * params.normalization;
    }
    histOut.bin[idx] += result;
}
