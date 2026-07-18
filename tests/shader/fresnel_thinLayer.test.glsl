layout(local_size_x = 64) in;

//N = 32 * 64
#define N 2048
#define SEED 0xC0FFEEu

#include "random/util.glsl"
#include "util/hash.glsl"

#ifdef INCLUDE_INTERFERENCE
#include "surface/thin/fresnel_complex.glsl"
#else
#include "surface/thin/fresnel_real.glsl"
#endif

writeonly buffer Result{
    float n1[N];
    float n2[N];
    float k2[N];
    float n3[N];
    float d[N];
    float cos0[N];
    float lambda[N];

    float Rs[N];
    float Rp[N];
    float Ts[N];
    float Tp[N];
} result;

float rand(uint i) {
    return normalizeUint(hash(gl_GlobalInvocationID.x, i, SEED));
}

void main() {
    //sample some random values
    float n1 = 1.0 + rand(0);
    float n2 = 1.0 + 2.0 * rand(1);
    float k2 = 2.5 * rand(2);
    float n3 = 1.0 + rand(3);
    float d = 10.0 + 90.0 * rand(4);
    float cos0 = 1.0 - rand(5);
    float lambda = 250.0 + 500.0 * rand(6);
    //randomly choose between dielectric and metallic layer
    k2 = rand(7) < 0.5 ? k2 : 0.0;

    //evaluate fresnel terms
    #ifdef INCLUDE_INTERFERENCE
    vec4 fresnel = fresnel_thinLayer(n1, n2, k2, n3, d, lambda, cos0);
    #else
    //metalls always require interference handling
    k2 = 0.0;
    vec4 fresnel = fresnel_thinLayer(n1, n2, n3, cos0);
    #endif

    //save inputs & outputs
    uint i = gl_GlobalInvocationID.x;
    result.n1[i] = n1;
    result.n2[i] = n2;
    result.k2[i] = k2;
    result.n3[i] = n3;
    result.d[i] = d;
    result.cos0[i] = cos0;
    result.lambda[i] = lambda;
    result.Rs[i] = fresnel.x;
    result.Rp[i] = fresnel.y;
    result.Ts[i] = fresnel.z;
    result.Tp[i] = fresnel.w;
}
