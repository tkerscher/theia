#ifndef _SCATTER_INCLUDE
#define _SCATTER_INCLUDE

#include "material.glsl"

#define TWO_PI 6.283185307179586477
#define INV_4PI 0.0795774715459476679

////////////////////////////////////// RAY /////////////////////////////////////

struct Ray {
    vec3 position;
    vec3 direction;
    float wavelength;

    float log_trans;    //log(transmission)
    float log_prob;     //log(probability)
    float travelTime;

    Medium medium;
    MediumConstants constants; // cache lookup
};

Ray initRay(
    vec3 position,
    vec3 direction,
    float wavelength,
    const Medium medium,
    float log_transmission,
    float delta_time
) {
    return Ray(
        position,
        direction,
        wavelength,
        log_transmission,
        0.0, //log(p) = log(1.0) = 0.0
        delta_time,
        medium,
        lookUpMedium(medium, wavelength)
    );
}
Ray initRay(
    vec3 position,
    vec3 direction,
    float wavelength,
    const Medium medium
) {
    return initRay(
        position,
        direction,
        wavelength,
        medium,
        0.0,        // log_transmission
        0.0         // log_prob
    );
}

//////////////////////////////// MEDIUM SCATTER ////////////////////////////////

vec3 scatterDir(vec3 prevDir, float cos_theta, float phi) {
    //sanitize just to be safe
    prevDir = normalize(prevDir);

    //construct scattered direction in prevDir system
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    vec3 localScattered = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    //just to be safe
    localScattered = normalize(localScattered);

    //build local coordinate system
    vec3 z = prevDir;
    //edge case: prevDir points in z
    vec3 x = prevDir.z >= 0.999 ?
        cross(vec3(0.0,1.0,0.0), z) :
        cross(vec3(0.0,1.0,0.0), z);
    vec3 y = cross(x, z);

    //normalize again (this might be paranoid)
    x = normalize(x);
    y = normalize(y);

    //build transformation matrix (each vec is a column)
    mat3 trafo = mat3(x, y, z);

    //transform to global coordinates and return
    return normalize(trafo * localScattered);
}

void scatterMedium(inout Ray ray, float cos_theta, float phi) {
    //update direction
    ray.direction = scatterDir(ray.direction, cos_theta, phi);

    //calculate transmission
    //note, that calling this function is only valid if ray.medium is not null
    //since null means a vacuum, this should never happen
    ray.log_trans += lookUp(ray.medium.log_phase, cos_theta, INV_4PI);
}

// // Importance sampling using the phase function
// void scatterMedium_ISPhase(inout Ray ray) {
//     // sample cos_theta
//     float cos_theta;
//     if (uint64_t(ray.medium.phase_sampling) != 0) {
//         cos_theta = lookUp(ray.medium.phase_sampling, rand());
//         cos_theta = clamp(cos_theta, -1.0, 1.0);
//     }
//     else {
//         cos_theta = 2.0 * rand() - 1.0;
//     }
//     // sample phi
//     float phi = TWO_PI * rand();

//     //scatter
//     ray.direction = scatterDir(ray.direction, cos_theta, phi);

//     //we don't have to update radiance and probability as they will exactly
//     //cancel out during the summation of the MC integral
// }

/////////////////////////////// MATERIAL SCATTER ///////////////////////////////

float reflectance(const Ray ray, const Material material, vec3 normal) {
    //calculate incidence angle
    float cos_i = clamp(dot(normalize(ray.direction), normal), -1.0, 1.0);
    float sin_i = sqrt(max(1.0 - cos_i*cos_i, 0.0));

    //fetch refractive index on both sides
    Medium otherMed = cos_i <= 0.0 ? material.inside : material.outside;
    float n_t = 1.0;
    if (uint64_t(otherMed) != 0) {
        float u_t = normalize_lambda(otherMed, ray.wavelength);
        n_t = lookUp(otherMed.n, u_t, 1.0);
    }
    float n_i = ray.constants.n;

    //calculate outgoing angle (Snell's law)
    float sin_t = sin_i * n_i / n_t;
    //by clamping cos_t to 0.0 we accurately handle total internal reflection
    float cos_t = sqrt(max(1.0 - sin_t*sin_t, 0.0));

    //for the reflectance we need the absolute value
    cos_i = abs(cos_i);
    //calculate reflectance (Fresnel equations)
    float r_per = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t);
    float r_par = (n_i * cos_t - n_t * cos_i) / (n_t * cos_i + n_i * cos_t);
    //return average (unpolarized light)
    return clamp(0.5 * (r_par*r_par + r_per*r_per), 0.0, 1.0);
}

void reflectMaterial(inout Ray ray, const Material mat, vec3 normal) {
    //decision made elsewhere, just update ray

    ray.log_trans += log(reflectance(ray, mat, normal));
    ray.direction = normalize(reflect(ray.direction, normal));
}

void transmitMaterial(inout Ray ray, const Material mat, vec3 normal) {
    //decision made elsewhere, just update ray

    //we only need the sign, so skip normalizing
    float cos_i = dot(ray.direction, normal);

    //look up constants
    Medium med_t = cos_i <= 0.0 ? mat.inside : mat.outside;
    MediumConstants const_t = lookUpMedium(med_t, ray.wavelength);

    //update ray
    float n_i = ray.constants.n;
    float n_t = const_t.n;
    ray.log_trans += log(1.0 - reflectance(ray, mat, normal));
    ray.medium = med_t;
    ray.constants = const_t;
    //flip the normal if we go outside
    //cos_i is negative if anti-parallel to normal
    normal *= -sign(cos_i);
    ray.direction = normalize(refract(ray.direction, normal, n_i / n_t));
}

// // Importance sample material / surface interaction using reflectivity
// void sampleMaterialIS(inout Ray ray, const Material mat, vec3 normal) {
//     //get reflectance
//     float r = reflectance(ray, mat, normal);

//     //flip the normal if we go outside
//     //cos_i is negative if anti-parallel to normal
//     normal *= -sign(dot(ray.direction, normal));

//     //coin flip: reflect or transmit?
//     if (rand() <= r) {
//         //reflect
//         ray.direction = normalize(reflect(ray.direction, normal));
//     }
//     else {
//         //transmit
        
//         //look up constants
//         Medium med_t =  <= 0.0 ? mat.inside : mat.outside;
//         MediumConstants const_t = lookUpMedium(med_t, ray.wavelength);

//         //update ray
//         float n_i = ray.constants.n;
//         float n_t = const_t.n;
//         ray.direction = normalize(refract(ray.direction, normal, n_i / n_t));
//         ray.medium = med_t;
//         ray.constants = const_t;
//     }
//     //we do NOT update transmission as it cancels with probability
// }

#endif
