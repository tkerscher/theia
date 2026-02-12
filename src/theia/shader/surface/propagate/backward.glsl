#ifndef _INCLUDE_SURFACE_PROPAGATE_BACKWARD
#define _INCLUDE_SURFACE_PROPAGATE_BACKWARD

#include "util/offset.glsl"

/**
 * Lets the ray cross the medium border at the given surface hit without any
 * transmission effects. Useful for volume borders.
*/
ResultCode crossBorder(
    inout BackwardRay ray,  ///< Ray to cross
    const SurfaceHit hit    ///< Description of surface hit
) {
    ray.position = offsetRay(hit.worldPos, -hit.rayNrm);
    return updateMedium(ray, hit.otherMediumIdx);
}

/**
 * Reflects the ray at the given surface hit into the given direction
*/
ResultCode reflectRay(
    inout BackwardRay ray,  ///< Ray to reflect
    const SurfaceHit hit,   ///< Description of surface hit
    vec3 newDir             ///< Direction in which to reflect
) {
    //offset ray to prevent self-intersection in next tracing step
    ray.position = offsetRay(hit.worldPos, hit.rayNrm);
    return reflectRay(ray, newDir, hit.otherMediumIdx);
}
/**
 * Specular reflection of the given ray at the surface hit using its surface normal
*/
ResultCode reflectRay(
    inout BackwardRay ray,  ///< Ray to reflect
    const SurfaceHit hit    ///< Description of surface hit
) {
    //offset ray to prevent self-intersection in next tracing step
    ray.position = offsetRay(hit.worldPos, hit.rayNrm);
    vec3 newDir = reflect(ray.direction, hit.rayNrm);
    return reflectRay(ray, newDir, hit.otherMediumIdx);
}

/**
 * Transmits the ray at the given surface hit into the given direction
*/
ResultCode transmitRay(
    inout BackwardRay ray,  ///< Ray to transmit
    const SurfaceHit hit,   ///< Description of surface hit
    float n_i, float n_o,   ///< Refractive indices of incident and outgoing medium
    vec3 newDir             ///< Direction in which to transmit
) {
    //offset ray to prevent self-intersection in next tracing step
    ray.position = offsetRay(hit.worldPos, -hit.rayNrm);

    //transmitting radiance takes an additional factor eta^2
    //(this stems from a change of variable in the integral over outgoing to incoming angles)
    float eta = n_i / n_o;
    ray.lin_contrib *= eta * eta;

    return transmitRay(ray, newDir, hit.otherMediumIdx);
}
/**
 * Transmits the ray at the given surface hit into the given direction
*/
ResultCode transmitRay(
    inout BackwardRay ray,  ///< Ray to transmit
    const SurfaceHit hit,   ///< Description of surface hit
    vec3 newDir             ///< Direction in which to transmit
) {
    //fetch refractive indices
    float lam_i = normalize_lambda(ray.mediumIdx, ray.wavelength);
    float lam_o = normalize_lambda(hit.otherMediumIdx, ray.wavelength);
    float n_i = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, lam_i, 1.0);
    float n_o = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, lam_o, 1.0);

    //transmit
    return transmitRay(ray, hit, n_i, n_o, newDir);
}
/**
 * Specular transmission of the ray at given surface hit using its surface normal
*/
ResultCode transmitRay(
    inout BackwardRay ray,  ///< Ray to transmit
    const SurfaceHit hit,   ///< Description of surface hit
    float n_i, float n_o    ///< Refractive indices of incident and outgoing medium
) {
    //offset ray to prevent self-intersection in next tracing step
    ray.position = offsetRay(hit.worldPos, -hit.rayNrm);
    vec3 newDir = refract(ray.direction, hit.rayNrm, n_i / n_o);

    //If we try to transmit the ray when we are in the region of total internal
    //reflection, refract() returns a zero vector -> notify
    //(Due to finite numerical precision, the edge to total internal reflection
    //is a bit fuzzy)
    if (newDir == vec3(0.0)) return ERROR_CODE_TOTAL_INTERNAL_REFLECTION;

    //transmitting radiance takes an additional factor eta^2
    //(this stems from a change of variable in the integral over outgoing to incoming angles)
    float eta = n_i / n_o;
    ray.lin_contrib *= eta * eta;

    return transmitRay(ray, newDir, hit.otherMediumIdx);
}
/**
 * Specular transmission of the ray at given surface hit using its surface normal
*/
ResultCode transmitRay(
    inout BackwardRay ray,  ///< Ray to transmit
    const SurfaceHit hit    ///< Description of surface hit
) {
    //fetch refractive indices
    float lam_i = normalize_lambda(ray.mediumIdx, ray.wavelength);
    float lam_o = normalize_lambda(hit.otherMediumIdx, ray.wavelength);
    float n_i = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, lam_i, 1.0);
    float n_o = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, lam_o, 1.0);

    //transmit
    return transmitRay(ray, hit, n_i, n_o);
}

#endif
