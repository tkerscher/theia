#ifndef _INCLUDE_SURFACE_MICROFACET_TROWBRIDGE_REITZ
#define _INCLUDE_SURFACE_MICROFACET_TROWBRIDGE_REITZ

#include "math.glsl"

struct MicrofacetParams {
    float alpha;
    vec3 tangent1;
    vec3 tangent2;
};

MicrofacetParams prepare_microfacet(vec3 rayDir, const SurfaceHit hit){
    MicrofacetParams params;
    params.alpha = loadMaterialConstant(ROUGHNESS_PARAMETER, hit.materialIdx);
    params.tangent1 = perpendicularTo(hit.rayNrm);
    params.tangent2 = crosser(hit.rayNrm, params.tangent1);
    return params;
}

//sampling of micro-facets from the Trowbridge-Reitz (GGX) distribution
vec3 sample_microfacet_normal(
    const MicrofacetParams params,
    vec3 rayDir,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {

    //sample angles (theta, phi)
    vec2 rdm = random2D(idx, dim);
    float tan2_theta = params.alpha * params.alpha * rdm.x / (1 - rdm.x);
    float phi = TWO_PI * rdm.y;

    float cos_theta = inversesqrt(1.0 + tan2_theta);
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));

    //rotate surface normal
    return cos_theta * hit.rayNrm
        + sin_theta * cos(phi) * params.tangent1
        + sin_theta * sin(phi) * params.tangent2;
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
