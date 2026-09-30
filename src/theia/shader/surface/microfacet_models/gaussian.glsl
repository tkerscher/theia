#ifndef _INCLUDE_SURFACE_MICROFACET_GAUSSIAN
#define _INCLUDE_SURFACE_MICROFACET_GAUSSIAN

#include "math.glsl"

/*
Sampling of micro-facets from a Gaussian slope-angle distribution. The facet polar angle theta
follows

    p(theta) ~ sin(theta) * exp(-theta^2 / (2*sigma^2)),   theta in [0, pi/2]

with sigma = ROUGHNESS_PARAMETER. This distribution has no closed-form inverse
CDF, so we sample theta by rejection. The target is factored into an analytically
invertible proposal times a bounded acceptance weight:

    sin(theta) * exp(-theta^2/2sigma^2) = [theta * exp(-theta^2/2sigma^2)] * [sin(theta)/theta]
                                           \_______ Rayleigh(sigma) _______/   \___ <= 1 ___/

The Rayleigh factor is sampled analytically, truncated to [0, pi/2] so the
proposal never leaves the domain, and accepted with probability sin(theta)/theta,
which is <= 1 on [0, pi/2]. The overall acceptance rate is >= 0.81 for any sigma.
*/

struct MicrofacetParams {
    float sigma;
    float c;        //truncation mass of the Rayleigh proposal on [0, pi/2]
    vec3 tangent1;
    vec3 tangent2;
};

MicrofacetParams prepare_microfacet(vec3 rayDir, const SurfaceHit hit){
    MicrofacetParams params;
    params.sigma = loadMaterialConstant(ROUGHNESS_PARAMETER, hit.materialIdx);
    params.c = 1.0 - exp(-(PI_OVER_TWO * PI_OVER_TWO) / (2.0 * params.sigma * params.sigma));
    params.tangent1 = perpendicularTo(hit.rayNrm);
    params.tangent2 = crosser(hit.rayNrm, params.tangent1);
    return params;
}

vec3 sample_microfacet_normal(
    const MicrofacetParams params,
    vec3 rayDir,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {

    //rejection sample theta from the truncated Rayleigh proposal; accept ~ sin(theta)/theta.
    //4 attempts give a fall-through probability < 1.3e-3 even in the worst case (sigma -> inf),
    //and effectively zero for realistic roughnesses.
    float theta = 0.0;
    for (int i = 0; i < 4; ++i) {
        vec2 rdm = random2D(idx, dim);
        //truncated Rayleigh inverse CDF
        theta = params.sigma * sqrt(-2.0 * log(1.0 - params.c * rdm.x));
        float accept = theta > 1e-6 ? sin(theta) / theta : 1.0;
        if (rdm.y < accept)
            break;
    }
    float phi = TWO_PI * random(idx, dim);

    //rotate surface normal
    return cos(theta) * hit.rayNrm
        + sin(theta) * cos(phi) * params.tangent1
        + sin(theta) * sin(phi) * params.tangent2;
}

float microfacet_accept_prob(
    const MicrofacetParams params,
    vec3 dirOut,
    vec3 microfacetNormal,
    const SurfaceHit hit,
    vec3 rayDir
) {
    //only acccept facets that are hit from the front
    return dot(rayDir, microfacetNormal) < 0.0 ? 1.0 : 0.0;
}

#endif
