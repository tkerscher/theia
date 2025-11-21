#ifndef _INCLUDE_RANDOM_GAMMA
#define _INCLUDE_RANDOM_GAMMA

//Function for sampling the gamma distribution based on
//J. Ericsson: "Gamma Random Variable Generation on GPUs using CUDA" (2024)
//Note that algorithms performing well on CPU might be worse on GPU

//Cheng (GA) algorithm
float sampleGamma(float alpha, uint idx, inout uint dim) {
    //GA is only applicable for alpha >= 1
    //we can rescale the distribution to alpha + 1 to handle the range [0,1]
    float scale = 1.0;
    if (alpha < 1.0) {
        scale = pow(random(idx, dim), 1.0 / alpha);
        alpha += 1.0;
    }

    //rejection sampling loop
    float a = sqrt(2.0 * alpha - 1.0);
    float b = alpha - log(4);
    float c = alpha + 1.0 / alpha;
    while(true) {
        vec2 u = random2D(idx, dim);
        float V = a * log(u.x / (1.0 - u.x));
        float X = alpha * exp(V);

        if (b + c*V - X >= log(u.x*u.x*u.y)) return scale * X;
    }
}

float sampleGamma(float alpha, float lambda, uint idx, inout uint dim) {
    return sampleGamma(alpha, idx, dim) / lambda;
}

#endif
