#ifndef _INCLUDE_RANDOM_GAMMA
#define _INCLUDE_RANDOM_GAMMA

//Function for sampling the gamma distribution based on
//J. Ericsson: "Gamma Random Variable Generation on GPUs using CUDA" (2024)
//Note that algorithms performing well on CPU might be worse on GPU

//Max number of rejection sampling iterations. Since the acceptance rate is at
//least ~68%, all of them get rejected with a chance of less than 1e-16
//in which case the last proposal is used. The resulting bias is negligible.
//sampleGamma always consumes GAMMA_MAX_ITERATIONS + 1 dimensions.
//Must match the value used on the Python side.
#ifndef GAMMA_MAX_ITERATIONS
#define GAMMA_MAX_ITERATIONS 32
#endif

//Cheng (GA) algorithm
float sampleGamma(float alpha, uint idx, inout uint dim) {
    //reserve a fixed range of dimensions: one for rescaling, one per iteration
    uint d = dim;
    dim += GAMMA_MAX_ITERATIONS + 1;

    //GA is only applicable for alpha >= 1
    //we can rescale the distribution to alpha + 1 to handle the range [0,1]
    float scale = 1.0;
    if (alpha < 1.0) {
        scale = pow(random_s(idx, d), 1.0 / alpha);
        alpha += 1.0;
    }

    //rejection sampling loop
    float a = inversesqrt(2.0 * alpha - 1.0); //1/lam with lam = sqrt(2alpha−1), Cheng (1977)
    float b = alpha - log(4.0);
    float c = alpha + 1.0 / a;
    float X;
    for (uint i = 1; i <= GAMMA_MAX_ITERATIONS; ++i) {
        vec2 u = random2D_s(idx, d + i);
        float V = a * log(u.x / (1.0 - u.x));
        X = alpha * exp(V);

        if (b + c*V - X >= log(u.x*u.x*u.y)) break;
    }
    return scale * X;
}

float sampleGamma(float alpha, float lambda, uint idx, inout uint dim) {
    return sampleGamma(alpha, idx, dim) / lambda;
}

#endif
