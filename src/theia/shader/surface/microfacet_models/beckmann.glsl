#ifndef _INCLUDE_SURFACE_MICROFACET_BECKMANN
#define _INCLUDE_SURFACE_MICROFACET_BECKMANN

#include "math.glsl"

/*
Micro-facet parameters that are constant across a single surface interaction,
i.e. do not depend on the sampled facet. They are computed once via
prepare_microfacet() and then passed into sample_microfacet_normal() and
check_microfacet() on every retry, so the loop-invariant roughness load and
tangent basis are not recomputed per iteration of the rejection loop.
*/
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

//sampling of micro-facets from the Beckmann distribution
vec3 sample_microfacet_normal(const MicrofacetParams params, vec3 rayDir, const SurfaceHit hit, uint idx, inout uint dim){

    //sample angles (theta, phi)
    vec2 rdm = random2D(idx, dim);
    float tan2_theta = - params.alpha * params.alpha * log(1.0 - rdm.x);
    float phi = TWO_PI * rdm.y;

    float cos_theta = inversesqrt(1.0 + tan2_theta);
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));

    //rotate surface normal
    return cos_theta * hit.rayNrm + sin_theta * cos(phi) * params.tangent1 + sin_theta * sin(phi) * params.tangent2;
}

//check that ray hits micro-facet from the front
bool check_microfacet(const MicrofacetParams params, vec3 dirOut, vec3 microfacetNormal, const SurfaceHit hit, vec3 rayDir, uint idx, inout uint dim){
    return (dot(rayDir, microfacetNormal) < 0.0);
}

#endif
