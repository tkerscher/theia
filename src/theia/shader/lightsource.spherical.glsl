#ifndef _INCLUDE_LIGHTSOURCE_SPHERICAL
#define _INCLUDE_LIGHTSOURCE_SPHERICAL

#include "math.glsl"

layout(scalar) uniform LightParams {
    vec3 position;
    float budget;
} lightParams;

SourceRay sampleLight(uint idx) {
    //sample direction
    vec2 u = random2D(idx, 0);
    float cos_theta = 2.0 * u.x - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    float phi = TWO_PI * u.y;
    vec3 rayDir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    //sample photon
    SourceSample photon = sampleSource(idx, 2);
    //apply budget
    photon.contrib *= lightParams.budget;

    //assemble source ray
    return createSourceRay(
        lightParams.position,
        rayDir,
        photon
    );
}

#endif
