#ifndef _INCLUDE_LIGHTSOURCE_SPHERICAL
#define _INCLUDE_LIGHTSOURCE_SPHERICAL

#include "math.glsl"

layout(scalar) uniform LightParams {
    vec3 position;
    float contrib;

    float t_min;
    float t_max;
} lightParams;

SourceRay sampleLight(uint idx, uint dim) {
    //sample direction
    vec2 u = random2D(idx, dim); dim += 2;
    float cos_theta = 2.0 * u.x - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    float phi = TWO_PI * u.y;
    vec3 rayDir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    //sample startTime
    float v = random(idx, dim); dim++;
    float startTime = mix(lightParams.t_min, lightParams.t_max, v);
    //sample photon
    WavelengthSample photon = sampleWavelength(idx, dim);
    //apply budget
    photon.contrib *= lightParams.contrib;

    //assemble source ray
    return createSourceRay(
        lightParams.position,
        rayDir,
        startTime,
        photon
    );
}

#endif
