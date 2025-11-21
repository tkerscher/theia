#ifndef _INCLUDE_UTIL_SAMPLE
#define _INCLUDE_UTIL_SAMPLE

#include "math.glsl"

/**
 * Helper function converting spherical coords to cartessian
*/
vec3 sphericalToCartessian(float phi, float cos_theta) {
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    return vec3(
        sin_theta * sin(phi),
        sin_theta * cos(phi),
        cos_theta
    );
}

/**
 * Samples a random direction in the cone in z direction with an opening angle
 * whose cosine is denoted by cos_opening.
*/
vec3 sampleDirectionCone(float cos_opening, vec2 rng) {
    float phi = TWO_PI * rng.x;
    float cos_theta = (1.0 - rng.y) + cos_opening * rng.y;
    return sphericalToCartessian(phi, cos_theta);
}

/**
 * Samples unit disk in the xy-plane at z=0.
*/
vec3 sampleUnitDisk(vec2 rng) {
    //Use concentric sampling as shown in Chapter A.5 in
    //"Physical Based Rendering" by M. Pharr et al.

    //convert rng to [-1,1]^2
    rng = 2.0 * rng - 1.0;
    //handle degeneracy
    if (rng.x == 0.0 && rng.y == 0.0)
        return vec3(0.0);
    
    float phi, r;
    if (abs(rng.x) > abs(rng.y)) {
        r = rng.x;
        phi = PI_OVER_FOUR * (rng.y / rng.x);
    }
    else {
        r = rng.y;
        phi = PI_OVER_TWO - PI_OVER_FOUR * (rng.x / rng.y);
    }

    return vec3(r * cos(phi), r * sin(phi), 0.0);
}

/**
 * Samples a random direction in the upper hemisphere along the positive z axis.
 * Excludes vectors in the xy-plane, i.e. z=0.
*/
vec3 sampleHemisphere(vec2 rng) {
    float phi = TWO_PI * rng.x;
    float cos_theta = 1.0 - rng.y; //exclude 0.0
    return sphericalToCartessian(phi, cos_theta);
}

/**
 * Samples a random direction in the upper hemisphere along the positive z axis
 * weighted by the cosine of the angle to the z axis.
 * Excludes vectors in the xy-plane, i.e. z=0.
*/
vec3 sampleHemisphereCosine(vec2 rng) {
    //Here we use Malley's Method by projecting a disk onto a hemisphere.
    //See PBRT 4th Ed., Ch. 13.6.3
    vec2 d = sampleUnitDisk(rng).xy;
    float z = sqrt(max(0, 1.0 - dot(d, d)));
    return vec3(d, z);
}
float sampleHemisphereCosinePdf(vec3 dir) {
    //cos(theta) / pi
    return INV_PI * dir.z;
}

/**
 * Samples a random point on the unit sphere.
*/
vec3 sampleUnitSphere(vec2 rng) {
    float phi = TWO_PI * rng.x;
    float cos_theta = 2.0 * rng.y - 1.0;
    return sphericalToCartessian(phi, cos_theta);
}

#endif
