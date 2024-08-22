#ifndef _SCATTER_SURFACE_INCLUDE
#define _SCATTER_SURFACE_INCLUDE

#include "math.glsl"
#include "material.glsl"

/**
 * Struct holding all necessary properties of a surface interaction needed for
 * reflecting and transmitting. Used to declutter higher level code.
*/
struct SurfaceReflectance {
    Material material;  ///< Material of the surface
    vec3 rayNormal;     ///< Surface normal as seen by ray
    bool inward;        ///< Direction of the incident ray

    float n_in;         ///< Refractive index on the incident side
    float n_tr;         ///< Refractive index on the transmitted side

    float r_s;          ///< Fresnel term for perpendicular polarization
    float r_p;          ///< Fresnel term for parallel polarization
};

SurfaceReflectance fresnelReflect(
    const Material material,
    const float wavelength,
    const float n_i,
    const vec3 direction,
    const vec3 normal
) {
    //calculate incidence angle
    float cos_i = clamp(dot(direction, normal), -1.0, 1.0);
    float sin_i = sqrt(max(1.0 - cos_i*cos_i, 0.0));
    //calculate rayNormal opposing the ray's direction
    vec3 rayNormal = normal * (-signBit(cos_i));

    //fetch refractive index on other side
    bool inward = cos_i <= 0.0;
    Medium otherMed = inward ? material.inside : material.outside;
    float n_t = 1.0;
    if (uint64_t(otherMed) != 0) {
        float u_t = normalize_lambda(otherMed, wavelength);
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
    return SurfaceReflectance(
        material, rayNormal, inward,
        n_i, n_t,
        r_s, r_p
    );
}

// float reflectance(
//     const Material material,
//     const float wavelength,
//     const float n_i,
//     const vec3 direction,
//     const vec3 normal
// ) {
//     //calculate incidence angle
//     float cos_i = clamp(dot(direction, normal), -1.0, 1.0);
//     float sin_i = sqrt(max(1.0 - cos_i*cos_i, 0.0));

//     //fetch refractive index on other side
//     Medium otherMed = cos_i <= 0.0 ? material.inside : material.outside;
//     float n_t = 1.0;
//     if (uint64_t(otherMed) != 0) {
//         float u_t = normalize_lambda(otherMed, wavelength);
//         n_t = lookUp(otherMed.n, u_t, 1.0);
//     }

//     //calculate outgoing angle (Snell's law)
//     float sin_t = sin_i * n_i / n_t;
//     //by clamping cos_t to 0.0 we accurately handle total internal reflection
//     float cos_t = sqrt(max(1.0 - sin_t*sin_t, 0.0));

//     //for the reflectance we need the absolute value
//     cos_i = abs(cos_i);
//     //calculate reflectance (Fresnel equations)
//     float r_per = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t);
//     float r_par = (n_i * cos_t - n_t * cos_i) / (n_t * cos_i + n_i * cos_t);
//     //return average (unpolarized light)
//     return clamp(0.5 * (r_par*r_par + r_per*r_per), 0.0, 1.0);
// }

#endif
