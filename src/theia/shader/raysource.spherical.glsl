#ifndef _INCLUDE_RAYSAMPLER_SPHERICAL
#define _INCLUDE_RAYSAMPLER_SPHERICAL

#include "math.glsl"

layout(scalar) uniform RayParams {
    vec3 position;
} rayParams;

RaySample sampleRay(uint idx, uint dim) {
    //sample direction
    vec2 u = random2D(idx, dim);
    float cos_theta = 2.0 * u.x - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    float phi = TWO_PI * u.y;
    vec3 rayDir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );

    //assemble ray
    return RaySample(
        rayParams.position,
        rayDir,
        //p(ray) = 1/4pi -> contrib = 1/p(ray) = 4pi
        FOUR_PI
    );
}

#endif
