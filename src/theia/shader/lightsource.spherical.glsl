#ifndef _INCLUDE_LIGHTSOURCE_SPHERICAL
#define _INCLUDE_LIGHTSOURCE_SPHERICAL

#include "math.glsl"

layout(scalar) uniform LightParams {
    vec3 position;
    float contribFwd;
    float contribBwd;

    float t_min;
    float t_max;
} lightParams;

SourceRay sampleLight(float wavelength, uint idx, inout uint dim) {
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
    //sample startTime
    float v = random(idx, dim);
    float startTime = mix(lightParams.t_min, lightParams.t_max, v);

    //assemble source ray
    return createSourceRay(
        lightParams.position,
        rayDir,
        startTime,
        lightParams.contribFwd
    );
}

SourceRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    uint idx, inout uint dim
) {
    //get direction
    vec3 direction = normalize(observer - lightParams.position);
    
    //sample start time
    float u = random(idx, dim);
    float startTime = mix(lightParams.t_min, lightParams.t_max, u);

    //calculate contribution
    float d = distance(observer, lightParams.position);
    float contrib = lightParams.contribBwd / (d*d);

    //assemble source ray
    return createSourceRay(
        lightParams.position,
        direction,
        startTime,
        contrib
    );
}

#endif
