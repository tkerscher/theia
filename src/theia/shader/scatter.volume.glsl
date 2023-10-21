#ifndef _SCATTER_VOLUME_INCLUDE
#define _SCATTER_VOLUME_INCLUDE

#include "cosy.glsl"
#include "ray.glsl"

#define TWO_PI 6.283185307179586477
#define INV_4PI 0.0795774715459476679

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
    mat3 trafo = createLocalCOSY(prevDir);

    //transform to global coordinates and return
    return normalize(trafo * localScattered);
}

vec3 scatter(inout Ray ray, vec2 rng, out float p) {
    //importance sample scattering phase function
    float phi = rng.x * TWO_PI;
    float cos_theta;
    Medium medium = Medium(ray.medium);
    if (uint64_t(medium.phase_sampling) != 0) {
        cos_theta = lookUp(medium.phase_sampling, rng.y);
        cos_theta = clamp(cos_theta, -1.0, 1.0);
        //look up propability (assume that phase_sampling implies log_phase table)
        p = exp(lookUp(medium.log_phase, 0.5 * (cos_theta + 1.0)));
    }
    else {
        cos_theta = 2.0 * rng.y - 1.0;
        //constant probability
        p = INV_4PI;
    }

    //scatter
    ray.direction = scatterDir(ray.direction, cos_theta, phi);

    //since we importance sampled the phase function they cancel,
    //leaving us with only the scattering coefficient

    //update photons
    for (int i = 0; i < N_PHOTONS; ++i) {
        ray.photons[i].T_lin *= ray.photons[i].constants.mu_s;
    }

    //return direction
    return ray.direction;
}

float scatterProb(const Ray ray, vec3 scatterDir) {
    //check if we can sample the scattering
    Medium medium = Medium(ray.medium);
    if (uint64_t(medium.log_phase) == 0) {
        //uniform scattering prob
        return INV_4PI;
    }

    //look up prob using scattered cos_theta
    float cos_theta = dot(ray.direction, scatterDir); //[-1,1]
    float log_p = lookUp(medium.log_phase, 0.5 * (cos_theta + 1.0));
    return exp(log_p);
}

#endif
