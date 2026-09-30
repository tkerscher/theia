#ifndef _INCLUDE_SURFACE_PROPAGATE_FORWARD
#define _INCLUDE_SURFACE_PROPAGATE_FORWARD

#include "util/offset.glsl"

#ifdef RAY_PORTAL
//Portal tables + portalReframe(). Included BEFORE the surface model so the
//transmit paths below can switch scene on a portal crossing.
#include "tracer/scene/portal.glsl"
#endif

/**
 * Lets the ray cross the medium border at the given surface hit without any
 * transmission effects. Useful for volume borders.
*/
ResultCode crossBorder(
    inout ForwardRay ray,   ///< Ray to cross
    const SurfaceHit hit    ///< Description of surface hit
) {
#ifdef RAY_PORTAL
    //portal surface: no world-space offset (portalReframe sets position from the
    //precise object-space hit); just swap the medium, then switch scene/frame.
    if ((hit.flags & MATERIAL_PORTAL_BIT) != 0) {
        ResultCode result = updateMedium(ray, hit.otherMediumIdx);
        if (result >= 0) portalReframe(ray, hit);
        return result;
    }
#endif
    ray.position = offsetRay(hit.worldPos, hit.rayNrm, -hit.rayOffset);
    return updateMedium(ray, hit.otherMediumIdx);
}

/**
 * Reflects the ray at the given surface hit into the given direction
*/
ResultCode reflectRay(
    inout ForwardRay ray,   ///< Ray to reflect
    const SurfaceHit hit,   ///< Description of surface hit
    vec3 newDir             ///< Direction in which to reflect
) {
    //offset ray to prevent self-intersection in next tracing step
    ray.position = offsetRay(hit.worldPos, hit.rayNrm, hit.rayOffset);
    return reflectRay(ray, newDir, hit.otherMediumIdx);
}
/**
 * Specular reflection of the given ray at the surface hit using its surface normal
*/
ResultCode reflectRay(
    inout ForwardRay ray,   ///< Ray to reflect
    const SurfaceHit hit    ///< Description of surface hit
) {
    //offset ray to prevent self-intersection in next tracing step
    ray.position = offsetRay(hit.worldPos, hit.rayNrm, hit.rayOffset);
    vec3 newDir = reflect(ray.direction, hit.rayNrm);
    return reflectRay(ray, newDir, hit.otherMediumIdx);
}

/**
 * Transmits the ray at the given surface hit into the given direction
*/
ResultCode transmitRay(
    inout ForwardRay ray,   ///< Ray to transmit
    const SurfaceHit hit,   ///< Description of surface hit
    vec3 newDir             ///< Ray in which to transmit
) {
#ifdef RAY_PORTAL
    //portal surface: a TRANSMITTED ray switches scene/frame (reflected rays don't,
    //reflectRay is untouched). No world-space offset - portalReframe sets position
    //from the precise object-space hit and re-expresses the (transmitted) direction
    //in the target frame.
    if ((hit.flags & MATERIAL_PORTAL_BIT) != 0) {
        ResultCode result = transmitRay(ray, newDir, hit.otherMediumIdx);
        if (result >= 0) portalReframe(ray, hit);
        return result;
    }
#endif
    //offset ray to prevent self-intersection in next tracing step
    ray.position = offsetRay(hit.worldPos, hit.rayNrm, -hit.rayOffset);
    return transmitRay(ray, newDir, hit.otherMediumIdx);
}
/**
 * Specular transmission of the ray at given surface hit using its surface normal
*/
ResultCode transmitRay(
    inout ForwardRay ray,   ///< Ray to transmit
    const SurfaceHit hit,   ///< Description of surface hit
    float n_i, float n_o    ///< Refractive index of incident and outgoing medium
) {
    vec3 newDir = refract(ray.direction, hit.rayNrm, n_i / n_o);
    //If we try to transmit the ray when we are in the region of total internal
    //reflection, refract() returns a zero vector -> notify
    //(Due to finite numerical precision, the edge to total internal reflection
    //is a bit fuzzy)
    if (newDir == vec3(0.0)) return ERROR_CODE_TOTAL_INTERNAL_REFLECTION;
    //route through the (portal-aware) overload so the scene switch lives in one place
    return transmitRay(ray, hit, newDir);
}
ResultCode transmitRay(
    inout ForwardRay ray,   ///< Ray to transmit
    const SurfaceHit hit    ///< Description of surface hit
) {
    //fetch refractive indices
    float n_i = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, ray.wavelength, 1.0);
    float n_o = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 1.0);

    //transmit
    return transmitRay(ray, hit, n_i, n_o);
}

#endif
