#extension GL_EXT_scalar_block_layout : require

//parameterization via specialization constants
layout(local_size_x_id = 1) in;
layout(constant_id = 2) const uint N_BINS = 100;

layout(scalar) readonly buffer HistogramIn {
    float histIn[];
};
layout(scalar) writeonly buffer HistogramOut {
    float histOut[];
};

layout(scalar) uniform Params{
    float norm;
    uint nHist;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= N_BINS)
        return;
    
    //add same bin accross all inputs and save in input
    float result = 0.0;
    uint i = idx;
    for (uint n = 0; n < nHist; ++n, i += N_BINS) {
        result += histIn[i] * norm;
    }
    histOut[idx] = result;
}
