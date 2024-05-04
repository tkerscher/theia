#ifndef _SCATTER_VOLUME_INCLUDE
#define _SCATTER_VOLUME_INCLUDE

#include "math.glsl"
#include "material.glsl"

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

float sampleScatterDir(
    const Medium medium, vec3 inDir, vec2 rng,
    out float cos_theta, out float phi
) {
    //importance sample scattering phase function
    phi = rng.x * TWO_PI;
    if (uint64_t(medium.phase_sampling) != 0) {
        cos_theta = lookUp(medium.phase_sampling, rng.y);
        cos_theta = clamp(cos_theta, -1.0, 1.0);
        //look up propability (assume that phase_sampling implies log_phase table)
        return exp(lookUp(medium.log_phase, 0.5 * (cos_theta + 1.0)));
    }
    else {
        cos_theta = 2.0 * rng.y - 1.0;
        //constant probability
        return INV_4PI;
    }
}

vec3 scatter(const Medium medium, vec3 inDir, vec2 rng, out float p) {
    float cos_theta, phi;
    p = sampleScatterDir(medium, inDir, rng, cos_theta, phi);
    //scatter
    return scatterDir(inDir, cos_theta, phi);
}

float scatterProb(const Medium medium, vec3 inDir, vec3 scatterDir) {
    //check if we can sample the scattering
    if (uint64_t(medium.log_phase) == 0) {
        //uniform scattering prob
        return INV_4PI;
    }

    //look up prob using scattered cos_theta
    float cos_theta = dot(inDir, scatterDir); //[-1,1]
    float log_p = lookUp(medium.log_phase, 0.5 * (cos_theta + 1.0));
    return exp(log_p);
}

#endif
