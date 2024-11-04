#ifndef _INCLUDE_UTIL_SAMPLE
#define _INCLUDE_UTIL_SAMPLE

#include "math.glsl"

/**
 * Samples a random direction in the upper hemisphere along the positive z axis.
 * Excludes vectors in the xy-plane, i.e. z=0.
*/
vec3 sampleHemisphere(vec2 rng) {
    float phi = TWO_PI * rng.x;
    float cos_theta = 1.0 - rng.y; //exclude 0.0
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    return vec3(
        sin_theta * sin(phi),
        sin_theta * cos(phi),
        cos_theta
    );
}

/**
 * Samples a random point on the unit sphere.
*/
vec3 sampleUnitSphere(vec2 rng) {
    float phi = TWO_PI * rng.x;
    float cos_theta = 2.0 * rng.y - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    return vec3(
        sin_theta * sin(phi),
        sin_theta * cos(phi),
        cos_theta
    );
}

#endif
