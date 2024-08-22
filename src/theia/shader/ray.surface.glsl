#ifndef _INCLUDE_RAY_SURFACE
#define _INCLUDE_RAY_SURFACE

#include "material.glsl"
#include "ray.glsl"
#include "scatter.surface.glsl"

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
    Material material,      ///< Material of the border
    vec3 rayNormal,         ///< Normal of the surface as seen by the ray
    bool inward             ///< Whether to cross the border in- or outwards
) {
    //push ray to the other side
    ray.position = offsetRay(ray.position, -rayNormal);

    //update medium & constants
    Medium medium = inward ? material.inside : material.outside;
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
    inout RayState ray,                 ///< Ray to reflect
    const SurfaceReflectance surface    ///< Description of surface
) {
    //update ray state
    ray.position = offsetRay(ray.position, surface.rayNormal);
    ray.direction = normalize(reflect(ray.direction, surface.rayNormal));
}
/**
 * Reflects the ray from the given surface and updates its state accordingly.
 * Unlike reflectRayIS this does apply the reflectance to the ray's contribution.
 *
 * NOTE: For polarized rays, this assumes its reference frame is perpendicular
 *       to the plane of incidence.
*/
void reflectRay(
    inout RayState ray,                 ///< Ray to reflect
    const SurfaceReflectance surface    ///< Description of surface
) {
    //reflect ray
    reflectRayIS(ray, surface);
    //apply reflectance
    float s = surface.r_s;
    float p = surface.r_p;
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
    inout RayState ray,                 ///< Ray to transmit
    const SurfaceReflectance surface    ///< Description of surface
) {
    //cross border
    crossBorder(ray, surface.material, surface.rayNormal, surface.inward);
    //calculate new direction
    float eta = surface.n_in / surface.n_tr;
    ray.direction = normalize(refract(ray.direction, surface.rayNormal, eta));
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
    const SurfaceReflectance surface
) {
    //transmit ray
    transmitRayIS(ray, surface);
    //apply transmittance
    float s = surface.r_s;
    float p = surface.r_p;
    ray.lin_contrib *= 1.0 - 0.5 * (s*s + p*p); // T = 1 - R
}

///////////////////////////////  Specialization ////////////////////////////////

void crossBorder(inout ForwardRay ray, Material material, vec3 rayNormal, bool inward) {
    crossBorder(ray.state, material, rayNormal, inward);
}
void crossBorder(inout BackwardRay ray, Material material, vec3 rayNormal, bool inward) {
    crossBorder(ray.state, material, rayNormal, inward);
}

#ifdef POLARIZATION

#include "polarization.glsl"

void reflectRayIS(
    inout PolarizedForwardRay ray,
    const SurfaceReflectance surface
) {
    //reflect ray
    reflectRayIS(ray.state, surface);

    //update stokes vector
    //assume polRef is in the plane of incidence
    ray.stokes = polarizerMatrix(surface.r_p, surface.r_s) * ray.stokes;
}
void reflectRay(
    inout PolarizedForwardRay ray,
    const SurfaceReflectance surface
) {
    reflectRay(ray.state, surface);
    ray.stokes = polarizerMatrix(surface.r_p, surface.r_s) * ray.stokes;
}

void reflectRayIS(
    inout PolarizedBackwardRay ray,
    const SurfaceReflectance surface
) {
    //reflect ray
    reflectRayIS(ray.state, surface);

    //update mueller matrix
    //assume polRef is in the plane of incidence
    ray.mueller = ray.mueller * polarizerMatrix(surface.r_p, surface.r_s);
}
void reflectRay(
    inout PolarizedBackwardRay ray,
    const SurfaceReflectance surface
) {
    reflectRay(ray.state, surface);
    ray.mueller = ray.mueller * polarizerMatrix(surface.r_p, surface.r_s);
}

void transmitRayIS(
    inout PolarizedForwardRay ray,
    const SurfaceReflectance surface
) {
    //transmit ray
    transmitRayIS(ray.state, surface);

    //forward rays transport importance, which requires an additional factor of
    //eta^-2. See PBRT or Veach' thesis
    float eta = surface.n_in / surface.n_tr;
    ray.state.lin_contrib /= eta * eta;

    //update stokes vector
    //assume polRef is in the plane of incidence
    float t_s = surface.r_s + 1.0;
    float t_p = (surface.r_p + 1.0) * eta;
    ray.stokes = polarizerMatrix(t_p, t_s) * ray.stokes;
}
void transmitRay(
    inout PolarizedForwardRay ray,
    const SurfaceReflectance surface
) {
    transmitRay(ray.state, surface);

    float eta = surface.n_in / surface.n_tr;
    ray.state.lin_contrib /= eta * eta;

    float t_s = surface.r_s + 1.0;
    float t_p = (surface.r_p + 1.0) * eta;
    ray.stokes = polarizerMatrix(t_p, t_s) * ray.stokes;
}

void transmitRayIS(
    inout PolarizedBackwardRay ray,
    const SurfaceReflectance surface
) {
    //transmit ray
    transmitRayIS(ray.state, surface);

    //update mueller matrix
    //assume polRef is in the plane of incidence
    float eta = surface.n_in / surface.n_tr;
    float t_s = surface.r_s + 1.0;
    float t_p = (surface.r_p + 1.0) * eta;
    ray.mueller = ray.mueller * polarizerMatrix(t_p, t_s);
}
void transmitRay(
    inout PolarizedBackwardRay ray,
    const SurfaceReflectance surface
) {
    transmitRay(ray.state, surface);

    float eta = surface.n_in / surface.n_tr;
    float t_s = surface.r_s + 1.0;
    float t_p = (surface.r_p + 1.0) * eta;
    ray.mueller = ray.mueller * polarizerMatrix(t_p, t_s);
}

#else //#ifdef POLARIZATION

void reflectRayIS(
    inout UnpolarizedForwardRay ray,
    const SurfaceReflectance surface
) {
    reflectRayIS(ray.state, surface);
}
void reflectRayIS(
    inout UnpolarizedBackwardRay ray,
    const SurfaceReflectance surface
) {
    reflectRayIS(ray.state, surface);
}

void reflectRay(
    inout UnpolarizedForwardRay ray,
    const SurfaceReflectance surface
) {
    reflectRay(ray.state, surface);
}
void reflectRay(
    inout UnpolarizedBackwardRay ray,
    const SurfaceReflectance surface
) {
    reflectRay(ray.state, surface);
}

void transmitRayIS(
    inout UnpolarizedForwardRay ray,
    const SurfaceReflectance surface
) {
    //transmit ray
    transmitRayIS(ray.state, surface);

    //forward rays transport importance, which requires an additional factor of
    //eta^-2. See PBRT or Veach' thesis
    float inv_eta = surface.n_tr / surface.n_in;
    ray.state.lin_contrib *= inv_eta * inv_eta;
}
void transmitRayIS(
    inout UnpolarizedBackwardRay ray,
    const SurfaceReflectance surface
) {
    transmitRayIS(ray.state, surface);
}

void transmitRay(
    inout UnpolarizedForwardRay ray,
    const SurfaceReflectance surface
) {
    //transmit ray
    transmitRay(ray.state, surface);

    //forward rays transport importance, which requires an additional factor of
    //eta^-2. See PBRT or Veach' thesis
    float inv_eta = surface.n_tr / surface.n_in;
    ray.state.lin_contrib *= inv_eta * inv_eta;
}
void transmitRay(
    inout UnpolarizedBackwardRay ray,
    const SurfaceReflectance surface
) {
    transmitRay(ray.state, surface);
}

#endif

#endif
