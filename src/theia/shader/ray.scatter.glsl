#ifndef _INCLUDE_RAY_SCATTER
#define _INCLUDE_RAY_SCATTER

#include "ray.glsl"
#include "ray.medium.glsl"
#include "scatter.volume.glsl"

/**
 * Scatters ray in the given direction assuming it was importance sampled, i.e.
 * the phase function will not be applied, only the scattering coefficient.
*/
void scatterRayIS(inout RayState ray, vec3 dir) {
    ray.direction = dir;
    ray.lin_contrib *= ray.constants.mu_s;
}

/**
 * Scatters ray in the given direction assuming it is arbitrary, i.e. this will
 * apply both the phase function and scattering coefficient.
*/
void scatterRay(inout RayState ray, vec3 dir) {
    float phase = scatterProb(getMedium(ray), ray.direction, dir);
    ray.lin_contrib *= ray.constants.mu_s * phase;
    ray.direction = dir;
}

/**
 * Scatters the ray into a random direction using the given random numbers.
 * Importance samples the phase function.
*/
void scatterRay(inout RayState ray, vec2 u) {
    //sample new direction
    float cos_theta, phi;
    sampleScatterDir(getMedium(ray), ray.direction, u, cos_theta, phi);
    vec3 newDir = scatterDir(ray.direction, cos_theta, phi);
    //scatter
    scatterRayIS(ray, newDir);
}

///////////////////////// Polarization Specialization //////////////////////////

#ifdef POLARIZATION

#include "polarization.glsl"

void _scatterPolRay_impl(inout PolarizedForwardRay ray, vec3 dir, float cos_theta) {
    vec3 polRef;
    mat4 phase = lookUpPhaseMatrix(getMedium(ray), cos_theta);
    //rotates rays polRef to plane of scattering
    mat4 rotate = rotatePolRef(ray.state.direction, ray.polRef, dir, polRef);

    ray.stokes = phase * rotate * ray.stokes;
    ray.polRef = polRef;
}
void _scatterPolRay_impl(inout PolarizedBackwardRay ray, vec3 dir, float cos_theta) {
    vec3 polRef;
    mat4 phase = lookUpPhaseMatrix(getMedium(ray), cos_theta);
    //rotates rays polRef to plane of scattering
    mat4 rotate = rotatePolRef(ray.state.direction, ray.polRef, dir, polRef);

    //remember that we treat the mueller matrix as if seen by a photon
    //thus the correct rotation is from -dir -> -ray.dir, which happens after
    //the scattering to align with the previously sampled, but for the photon's
    //point of view next event. Mathematically, this is the inverse matrix which
    //here is its transpose since the rotation matrix is orthogonal.
    ray.mueller = ray.mueller * transpose(rotate) * phase;
    ray.polRef = polRef;
}

void scatterRayIS(inout PolarizedForwardRay ray, vec3 dir) {
    float cos_theta = dot(ray.state.direction, dir);
    _scatterPolRay_impl(ray, dir, cos_theta);
    scatterRayIS(ray.state, dir);
}
void scatterRayIS(inout PolarizedBackwardRay ray, vec3 dir) {
    float cos_theta = dot(ray.state.direction, dir);
    _scatterPolRay_impl(ray, dir, cos_theta);
    scatterRayIS(ray.state, dir);
}

void scatterRay(inout PolarizedForwardRay ray, vec3 dir) {
    float cos_theta = dot(ray.state.direction, dir);
    _scatterPolRay_impl(ray, dir, cos_theta);
    scatterRay(ray.state, dir);
}
void scatterRay(inout PolarizedBackwardRay ray, vec3 dir) {
    float cos_theta = dot(ray.state.direction, dir);
    _scatterPolRay_impl(ray, dir, cos_theta);
    scatterRay(ray.state, dir);
}

void scatterRay(inout PolarizedForwardRay ray, vec2 u) {
    //sample new direction
    float cos_theta, phi;
    sampleScatterDir(getMedium(ray), ray.state.direction, u, cos_theta, phi);
    vec3 newDir = scatterDir(ray.state.direction, cos_theta, phi);

    _scatterPolRay_impl(ray, newDir, cos_theta);
    scatterRayIS(ray.state, newDir);
}
void scatterRay(inout PolarizedBackwardRay ray, vec2 u) {
    //sample new direction
    float cos_theta, phi;
    sampleScatterDir(getMedium(ray), ray.state.direction, u, cos_theta, phi);
    vec3 newDir = scatterDir(ray.state.direction, cos_theta, phi);

    _scatterPolRay_impl(ray, newDir, cos_theta);
    scatterRayIS(ray.state, newDir);
}

#else //#ifdef POLARIZATION

void scatterRayIS(inout UnpolarizedForwardRay ray, vec3 dir) {
    scatterRayIS(ray.state, dir);
}
void scatterRayIS(inout UnpolarizedBackwardRay ray, vec3 dir) {
    scatterRayIS(ray.state, dir);
}

void scatterRay(inout UnpolarizedForwardRay ray, vec3 dir) {
    scatterRay(ray.state, dir);
}
void scatterRay(inout UnpolarizedBackwardRay ray, vec3 dir) {
    scatterRay(ray.state, dir);
}

void scatterRay(inout UnpolarizedForwardRay ray, vec2 u) {
    scatterRay(ray.state, u);
}
void scatterRay(inout UnpolarizedBackwardRay ray, vec2 u) {
    scatterRay(ray.state, u);
}

#endif //#ifdef POLARIZATION

#endif
