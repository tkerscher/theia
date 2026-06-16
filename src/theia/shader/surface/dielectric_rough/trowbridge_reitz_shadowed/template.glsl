#define SURFACE_MODEL_TROWBRIDGE_REITZ_SHADOWED

#include "math.glsl"
#include "util/sample.glsl"

/*
Sampling of visible micro-facets according to the Trowbeidge-Reitz model. The sampling algorithm
is described in [1].

[1] Matt Pharr, Wenzel Jakob, and Greg Humphreys "Physically Based Rendering: From Theory To Implementation"
    (2023) https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory
*/
vec3 sample_microfacet_normal(const RAY ray, const SurfaceHit hit, uint idx, inout uint dim){

    //Transformation matrix for local coordinate system. The surface normal is the new z-axis, and the
    //tangential component of the incoming ray is along the new y-axis.
    mat3 trafo = createLocalCOSY(hit.rayNrm, perpendicularTo(hit.rayNrm, ray.direction));

    float cos_n = abs(dot(hit.rayNrm, ray.direction));
    float alpha = loadMaterialSlot_vec2(ROUGHNESS_PARAMETER, hit.materialIdx).y;

    //transform surface normal to hemispherical configuration, using (0, -sin_n, -cos_n) as incoming direction
    vec3 wh = normalize(vec3(0.0, alpha * sqrt(max(1.0 - cos_n*cos_n, 0.0)), cos_n));

    //contruct orthonormal basis such that T1 is perpenticular to the surface normal
    vec3 T1 = vec3(-1.0, 0.0, 0.0);
    vec3 T2 = crosser(wh, T1);

    //sample point on unit disk
    vec3 p = sampleUnitDisk(random2D(idx, dim));

    //warp hemispherical projection for visible normal sampling
    float h = sqrt(1 - p.x*p.x);
    p.y = mix(h, p.y, (1.0 + wh.z) / 2.0);

    //reproject to hemisphere and transform normal to ellipsoid configuration
    p.z = sqrt(max(0.0, 1.0 - dot(p,p)));
    vec3 nh = p.x * T1 + p.y * T2 + p.z * wh;
    vec3 microfacetNormal_local = normalize(vec3(alpha * nh.x, alpha * nh.y, max(1e-6, nh.z)));

    //transform from local to global coordinate system
    return trafo * microfacetNormal_local;
}


//masking of outgoing rays
float masking_function(float cos_n, float alpha){
    //handle very small cosines
    if(cos_n < 1e-3){
        return 0;
    }

    float tan2_n = max(1.0 - cos_n*cos_n, 0.0) / (cos_n*cos_n);
    float Lambda = (sqrt(1.0 + alpha*alpha * tan2_n) - 1.0) / 2.0;
    return 1.0 / (1.0 + Lambda);
}


//rejection sample the masking
bool check_microfacet(vec3 dirOut, vec3 microfacetNormal, const SurfaceHit hit, const RAY ray, uint idx, inout uint dim){
    float u = random(idx, dim);
    float alpha = loadMaterialSlot_vec2(ROUGHNESS_PARAMETER, hit.materialIdx).y;
    float mask = masking_function(abs(dot(dirOut, hit.rayNrm)), alpha);
    return (mask > u);
}