#ifndef _INCLUDE_LIGHTSOURCE_SCATTER
#define _INCLUDE_LIGHTSOURCE_SCATTER

#ifdef POLARIZATION

#include "polarization.glsl"

void scatterSourceRay(
    inout SourceRay ray,
    uint mediumIdx,
    float mu_s,
    vec3 dir
) {
    float cos_theta = dot(ray.direction, dir);
    mat4 phase = lookUpPhaseMatrix(mediumIdx, cos_theta);
    //rotate polRef to plane of scattering
    vec3 polRef;
    mat4 rotate = rotatePolRef(ray.direction, ray.polRef, dir, polRef);

    //update stokes
    ray.stokes = phase * rotate * ray.stokes;
    ray.polRef = polRef;
    //update source ray
    ray.contrib *= mu_s * scatterProb(mediumIdx, ray.direction, dir);
    ray.direction = dir;
}

#else

void scatterSourceRay(
    inout SourceRay ray,
    uint mediumIdx,
    float mu_s,
    vec3 dir
) {
    //update source ray
    ray.contrib *= mu_s * scatterProb(mediumIdx, ray.direction, dir);
    ray.direction = dir;
}

#endif

#endif
