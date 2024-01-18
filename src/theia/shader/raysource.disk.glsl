#ifndef _INCLUDE_RAYSAMPLER_DISK
#define _INCLUDE_RAYSAMPLER_DISK

#include "cosy.glsl"
#include "math.glsl"

layout(scalar) uniform RayParams {
    vec3 center;
    vec3 direction;
    float radius;
} rayParams;

RaySample sampleRay(uint idx, uint dim) {
    //sample point on disk
    vec2 u = random2D(idx, dim);
    float r = rayParams.radius * sqrt(u.x);
    float phi = TWO_PI * u.y;
    vec3 pos = vec3(r*cos(phi), r*sin(phi), 0.0);
    //turn disk from z to final direction
    pos = createLocalCOSY(rayParams.direction) * pos;
    //offset disk
    pos += rayParams.center;

    //assemble ray and finish
    return RaySample(
        pos, rayParams.direction,
        //p(ray) = 1/A -> 1/p = pi*r^2
        PI * rayParams.radius * rayParams.radius
    );
}

#endif
