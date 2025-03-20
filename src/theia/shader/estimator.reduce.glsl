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
   //use compensated Kahan summation to reduce error
   //See https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    uint i = idx;
    precise float result = 0.0;
    precise float c = 0.0;
    for (uint n = 0; n < nHist; ++n, i += N_BINS) {
        precise float y = histIn[i] - c;
        precise float t = result + y;
        c = (t - result) - y;
        result = t;
    }
    histOut[idx] = result * norm;
}
