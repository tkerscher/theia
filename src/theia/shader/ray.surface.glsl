#ifndef _INCLUDE_RAY_SURFACE
#define _INCLUDE_RAY_SURFACE

#include "material.glsl"
#include "ray.glsl"
#include "scatter.surface.glsl"
#include "scene.types.glsl"

/**
 * Offsets ray position from surface hits to prevent self-intersection, i.e.
 * this ensures that after transmission/reflection the ray is actually on the
 * correct side of the geometry by compensating for the finite numerical
 * precision while introducing minimal bias.
 *
 * Normal (n) points outwards for rays existing the surface, else is flipped.
 * Offsets in normal direction.
 *
 * Taken from Ray Tracing Gems: Chapter 6
 * C. Waechter and N. Binder (2019): "A Fast and Robust Method for Avoiding
 * Self-Intersection"
*/
vec3 offsetRay(vec3 p, vec3 n) {
    ivec3 of_i = ivec3(256.0 * n);
    
    vec3 p_i = vec3(
        intBitsToFloat(floatBitsToInt(p.x)+((p.x < 0.0) ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(p.y)+((p.y < 0.0) ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(p.z)+((p.z < 0.0) ? -of_i.z : of_i.z))
    );

    return vec3(
        abs(p.x) < (1.0 / 32.0) ? p.x+ (1.0/65536.0)*n.x : p_i.x,
        abs(p.y) < (1.0 / 32.0) ? p.y+ (1.0/65536.0)*n.y : p_i.y,
        abs(p.z) < (1.0 / 32.0) ? p.z+ (1.0/65536.0)*n.z : p_i.z
    );
}

/**
 * Pushes ray across border and updates its state accordingly.
 * Applies an offset to prevent self-intersection of geometries.
 * Note that the surface normal is defined to always point outwards giving the
 * border its orientation.
*/
void crossBorder(
    inout RayState ray,     ///< Ray to update
    const SurfaceHit hit    ///< Hit describing the border
) {
    //push ray to the other side
    ray.position = offsetRay(hit.worldPos, -hit.rayNrm);

    //update medium & constants
    Medium medium = hit.inward ? hit.material.inside : hit.material.outside;
    ray.medium = uvec2(medium);
    ray.constants = lookUpMedium(medium, ray.wavelength);
}

/**
 * Reflects the ray from the given surface and updates it accordingly.
 * Assumes the decision whether to reflect was done using importance sampling,
 * thus reflectance will not be applied.
 *
 * NOTE: For polarized rays, this assumes its reference frame is perpendicular
 *       to the plane of incidence.
*/
void reflectRayIS(
    inout RayState ray,     ///< Ray to reflect
    const SurfaceHit hit,   ///< Hit description
    const Reflectance refl  ///< Description of surface
) {
    //update ray state
    ray.position = offsetRay(hit.worldPos, hit.rayNrm);
    ray.direction = normalize(reflect(ray.direction, hit.rayNrm));
}
/**
 * Reflects the ray from the given surface and updates its state accordingly.
 * Unlike reflectRayIS this does apply the reflectance to the ray's contribution.
 *
 * NOTE: For polarized rays, this assumes its reference frame is perpendicular
 *       to the plane of incidence.
*/
void reflectRay(
    inout RayState ray,     ///< Ray to reflect
    const SurfaceHit hit,   ///< Hit description
    const Reflectance refl  ///< Description of surface
) {
    //reflect ray
    reflectRayIS(ray, hit, refl);
    //apply reflectance
    float s = refl.r_s;
    float p = refl.r_p;
    ray.lin_contrib *= 0.5 * (s*s + p*p);
}

/**
 * Transmits the ray through the given surface and updates it accordingly.
 * Assumes the decision whether to transmit was done using importance sampling
 * thus transmittance will not be applied.
 *
 * NOTE: For polarized rays, this assumes its reference frame is perpendicular
 *       to the plane of incidence.
*/
void transmitRayIS(
    inout RayState ray,     ///< Ray to transmit
    const SurfaceHit hit,   ///< Hit description
    const Reflectance refl  ///< Description of surface
) {
    //cross border
    crossBorder(ray, hit);
    //calculate new direction
    float eta = refl.n_in / refl.n_tr;
    ray.direction = normalize(refract(ray.direction, hit.rayNrm, eta));
}
/**
 * Transmits the ray through the given surface and updates it accordingly.
 * Unlike transmitRayIS this does apply the transmittance to the ray's
 * contribution.
 *
 * NOTE: For polarized rays, this assumes its reference frame is perpendicular
 *       to the plane of incidence.
*/
void transmitRay(
    inout RayState ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    //transmit ray
    transmitRayIS(ray, hit, refl);
    //apply transmittance
    float s = refl.r_s;
    float p = refl.r_p;
    ray.lin_contrib *= 1.0 - 0.5 * (s*s + p*p); // T = 1 - R
}

///////////////////////////////  Specialization ////////////////////////////////

void crossBorder(inout ForwardRay ray, const SurfaceHit hit) {
    crossBorder(ray.state, hit);
}
void crossBorder(inout BackwardRay ray, const SurfaceHit hit) {
    crossBorder(ray.state, hit);
}

#ifdef POLARIZATION

#include "polarization.glsl"

void reflectRayIS(
    inout PolarizedForwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    //reflect ray
    reflectRayIS(ray.state, hit, refl);

    //update stokes vector
    //assume polRef is in the plane of incidence
    ray.stokes = polarizerMatrix(refl.r_p, refl.r_s) * ray.stokes;
}
void reflectRay(
    inout PolarizedForwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    reflectRay(ray.state, hit, refl);
    ray.stokes = polarizerMatrix(refl.r_p, refl.r_s) * ray.stokes;
}

void reflectRayIS(
    inout PolarizedBackwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    //reflect ray
    reflectRayIS(ray.state, hit, refl);

    //update mueller matrix
    //assume polRef is in the plane of incidence
    ray.mueller = ray.mueller * polarizerMatrix(refl.r_p, refl.r_s);
}
void reflectRay(
    inout PolarizedBackwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    reflectRay(ray.state, hit, refl);
    ray.mueller = ray.mueller * polarizerMatrix(refl.r_p, refl.r_s);
}

void transmitRayIS(
    inout PolarizedForwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    //transmit ray
    transmitRayIS(ray.state, hit, refl);

    //forward rays transport importance, which cancel the factor eta^2
    //See PBRT or Veach' thesis

    //update stokes vector
    //assume polRef is in the plane of incidence
    float eta = refl.n_in / refl.n_tr;
    float t_s = refl.r_s + 1.0;
    float t_p = (refl.r_p + 1.0) * eta;
    ray.stokes = polarizerMatrix(t_p, t_s) * ray.stokes;
}
void transmitRay(
    inout PolarizedForwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    transmitRay(ray.state, hit, refl);

    float eta = refl.n_in / refl.n_tr;
    float t_s = refl.r_s + 1.0;
    float t_p = (refl.r_p + 1.0) * eta;
    ray.stokes = polarizerMatrix(t_p, t_s) * ray.stokes;
}

void transmitRayIS(
    inout PolarizedBackwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    //transmit ray
    transmitRayIS(ray.state, hit, refl);

    //transmitting radiance takes an additional factor eta^2
    float eta = refl.n_in / refl.n_tr;
    ray.state.lin_contrib *= eta * eta;

    //update mueller matrix
    //assume polRef is in the plane of incidence
    float t_s = refl.r_s + 1.0;
    float t_p = (refl.r_p + 1.0) * eta;
    ray.mueller = ray.mueller * polarizerMatrix(t_p, t_s);
}
void transmitRay(
    inout PolarizedBackwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    transmitRay(ray.state, hit, refl);

    float eta = refl.n_in / refl.n_tr;
    ray.state.lin_contrib *= eta * eta;

    float t_s = refl.r_s + 1.0;
    float t_p = (refl.r_p + 1.0) * eta;
    ray.mueller = ray.mueller * polarizerMatrix(t_p, t_s);
}

#else //#ifdef POLARIZATION

void reflectRayIS(
    inout UnpolarizedForwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    reflectRayIS(ray.state, hit, refl);
}
void reflectRayIS(
    inout UnpolarizedBackwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    reflectRayIS(ray.state, hit, refl);
}

void reflectRay(
    inout UnpolarizedForwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    reflectRay(ray.state, hit, refl);
}
void reflectRay(
    inout UnpolarizedBackwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    reflectRay(ray.state, hit, refl);
}

void transmitRayIS(
    inout UnpolarizedForwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    //transmit ray
    transmitRayIS(ray.state, hit, refl);

    //forward rays transport importance, which cancel the factor eta^2
    //See PBRT or Veach' thesis
}
void transmitRayIS(
    inout UnpolarizedBackwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    transmitRayIS(ray.state, hit, refl);

    //transmitting radiance takes an additional factor eta^2
    float eta = refl.n_in / refl.n_tr;
    ray.state.lin_contrib *= eta * eta;
}

void transmitRay(
    inout UnpolarizedForwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    //transmit ray
    transmitRay(ray.state, hit, refl);
}
void transmitRay(
    inout UnpolarizedBackwardRay ray,
    const SurfaceHit hit,
    const Reflectance refl
) {
    transmitRay(ray.state, hit, refl);

    float eta = refl.n_in / refl.n_tr;
    ray.state.lin_contrib *= eta * eta;
}

#endif

#endif
