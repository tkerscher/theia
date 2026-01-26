#ifndef _INCLUDE_VOLUME_SCATTER
#define _INCLUDE_VOLUME_SCATTER

#include "math.glsl"

/**
 * Calculates scattered direction based on the previous direction using
 * angular coordinates cos(theta) and phi. Note that while it is deterministic
 * in phi, i.e. same values produces same results, its exact mapping is not
 * guaranteed. 
*/
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

/**
 * Samples the given media's phase function and returns both the sampled
 * scattering angles as well as the corresponding sample probability.
*/
float sampleScatterDir(
    const uint mediumIdx, vec3 inDir, vec2 rng,
    out float cos_theta, out float phi
) {
    //fetch corresponding slots
    Table1D phase_sampling = loadMediaSlot_Table1D(PHASE_SAMPLING, mediumIdx);
    Table1D log_phase = loadMediaSlot_Table1D(LOG_PHASE_FUNCTION, mediumIdx);
    //importance sample scattering phase function
    phi = rng.x * TWO_PI;
    if (uint64_t(phase_sampling) != 0) {
        cos_theta = lookUp(phase_sampling, rng.y);
        cos_theta = clamp(cos_theta, -1.0, 1.0);
        //look up propability (assume that phase_sampling implies log_phase table)
        return exp(lookUp(log_phase, 0.5 * (cos_theta + 1.0)));
    }
    else {
        cos_theta = 2.0 * rng.y - 1.0;
        //constant probability
        return INV_4PI;
    }
}

/**
 * Samples the given media's phase function and returns the scattered direction
 * based on the unscattered direction.
*/
vec3 scatter(const uint mediumIdx, vec3 inDir, vec2 rng, out float p) {
    float cos_theta, phi;
    p = sampleScatterDir(mediumIdx, inDir, rng, cos_theta, phi);
    //scatter
    return scatterDir(inDir, cos_theta, phi);
}

/**
 * Calculates the probability that the initial direction is scattered in the
 * scattered direction by the medium. Assumes that the medium scatters, i.e.
 * mu_s != 0!
*/
float scatterProb(const uint mediumIdx, vec3 inDir, vec3 scatterDir) {
    Table1D log_phase = loadMediaSlot_Table1D(LOG_PHASE_FUNCTION, mediumIdx);
    //check if we can sample the scattering
    if (uint64_t(log_phase) == 0) {
        //uniform scattering prob
        return INV_4PI;
    }

    //look up prob using scattered cos_theta
    float cos_theta = dot(inDir, scatterDir); //[-1,1]
    float log_p = lookUp(log_phase, 0.5 * (cos_theta + 1.0));
    return exp(log_p);
}

#endif
