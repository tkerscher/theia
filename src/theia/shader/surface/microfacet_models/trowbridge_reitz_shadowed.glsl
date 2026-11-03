#ifndef _INCLUDE_SURFACE_MICROFACET_TROWBRIDGE_REITZ_SHADOWED
#define _INCLUDE_SURFACE_MICROFACET_TROWBRIDGE_REITZ_SHADOWED

#include "math.glsl"
#include "util/sample.glsl"

/*
Sampling of visible micro-facets according to the Trowbridge-Reitz model. The sampling algorithm
is described in [1].

[1] Matt Pharr, Wenzel Jakob, and Greg Humphreys "Physically Based Rendering: From Theory To Implementation"
    (2023) https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory
*/

struct MicrofacetParams {
    float alpha;
    mat3 trafo;     //local coordinate system (surface normal = z-axis)
    vec3 wh;        //stretched incoming direction in hemispherical configuration
    vec3 T1;        //orthonormal basis for the visible-normal disk sampling
    vec3 T2;
};

MicrofacetParams prepareMicrofacet(vec3 rayDir, const SurfaceHit hit){
    MicrofacetParams params;
    params.alpha = loadMaterialConstant(ROUGHNESS_PARAMETER, hit.materialIdx);

    //Transformation matrix for local coordinate system. The surface normal is the new z-axis, and the
    //tangential component of the incoming ray is along the new y-axis.
    params.trafo = createLocalCOSY(hit.rayNrm, perpendicularTo(hit.rayNrm, rayDir));

    float cos_n = abs(dot(hit.rayNrm, rayDir));
    //transform surface normal to hemispherical configuration, using (0, -sin_n, -cos_n) as incoming direction
    params.wh = normalize(vec3(0.0, params.alpha * sqrt(max(1.0 - cos_n*cos_n, 0.0)), cos_n));

    //contruct orthonormal basis such that T1 is perpenticular to the surface normal
    params.T1 = vec3(-1.0, 0.0, 0.0);
    params.T2 = crosser(params.wh, params.T1);
    return params;
}

vec3 sampleMicrofacetNormal(
    const MicrofacetParams params,
    vec3 rayDir,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {

    //sample point on unit disk
    vec3 p = sampleUnitDisk(random2D(idx, dim));

    //warp hemispherical projection for visible normal sampling
    float h = sqrt(1 - p.x*p.x);
    p.y = mix(h, p.y, (1.0 + params.wh.z) / 2.0);

    //reproject to hemisphere and transform normal to ellipsoid configuration
    p.z = sqrt(max(0.0, 1.0 - dot(p,p)));
    vec3 nh = p.x * params.T1 + p.y * params.T2 + p.z * params.wh;
    vec3 microfacetNormal_local = normalize(vec3(
        params.alpha * nh.x, params.alpha * nh.y, max(1e-6, nh.z)));

    //transform from local to global coordinate system
    return params.trafo * microfacetNormal_local;
}

//masking of outgoing rays
float maskingFunction(float cos_n, float alpha){
    //handle very small cosines
    if(cos_n < 1e-3){
        return 0.0;
    }

    float tan2_n = max(1.0 - cos_n*cos_n, 0.0) / (cos_n*cos_n);
    float Lambda = (sqrt(1.0 + alpha*alpha * tan2_n) - 1.0) / 2.0;
    return 1.0 / (1.0 + Lambda);
}

float microfacetAcceptProb(
    const MicrofacetParams params,
    vec3 dirOut,
    vec3 microfacetNormal,
    const SurfaceHit hit,
    vec3 rayDir
) {
    //acceptance probability is given by the masking function
    return maskingFunction(abs(dot(dirOut, hit.rayNrm)), params.alpha);
}

#endif
