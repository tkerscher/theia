#ifndef _SCATTER_SURFACE_INCLUDE
#define _SCATTER_SURFACE_INCLUDE

#include "math.glsl"
#include "material.glsl"
#include "ray.glsl"
#include "scene.types.glsl"

/**
 * Struct holding all necessary properties of a surface interaction needed for
 * reflecting and transmitting. Used to declutter higher level code.
*/
struct Reflectance {
    float n_in;         ///< Refractive index on the incident side
    float n_tr;         ///< Refractive index on the transmitted side

    float r_s;          ///< Fresnel term for perpendicular polarization
    float r_p;          ///< Fresnel term for parallel polarization
};

Reflectance fresnelReflect(const RayState ray, const SurfaceHit hit) {
    //calculate incidence angle
    float cos_i = clamp(dot(ray.direction, hit.rayNrm), -1.0, 1.0);
    float sin_i = sqrt(max(1.0 - cos_i*cos_i, 0.0));

    //fetch refractive index on other side
    Medium otherMed = hit.inward ? hit.material.inside : hit.material.outside;
    float n_i = ray.constants.n;
    float n_t = 1.0;
    if (uint64_t(otherMed) != 0) {
        float u_t = normalize_lambda(otherMed, ray.wavelength);
        n_t = lookUp(otherMed.n, u_t, 1.0);
    }

    //calculate outgoing angle (Snell's law)
    float sin_t = sin_i * n_i / n_t;
    //by clamping cos_t to 0.0 we accurately handle total internal reflection
    float cos_t = sqrt(max(1.0 - sin_t*sin_t, 0.0));

    //for the reflectance we need the absolute value
    cos_i = abs(cos_i);
    //calculate reflectance (Fresnel equations)
    float r_s = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t);
    float r_p = (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t);

    //assemble result
    return Reflectance(
        n_i, n_t,
        r_s, r_p
    );
}

#endif
