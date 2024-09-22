#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV_SIMPLE
#define _INCLUDE_LIGHTSOURCE_CHERENKOV_SIMPLE

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

#include "lightsource.cherenkov.common.glsl"
#include "math.glsl"
#include "material.glsl"

layout(scalar) uniform LightParams {
    vec3 trackStart;
    float startTime;

    vec3 trackEnd;
    float endTime;

    vec3 trackDir;
    float trackDist;

    uvec2 medium;
} lightParams;

SourceRay sampleLight(float wavelength, uint idx, inout uint dim) {
    //interpolate track
    float u = random(idx, dim);
    vec3 position = mix(lightParams.trackStart, lightParams.trackEnd, u);
    float startTime = mix(lightParams.startTime, lightParams.endTime, u);

    //get refractive index
    float n = 1.0;
    if (lightParams.medium != uvec2(0)) {
        Medium medium = Medium(lightParams.medium);
        float l = normalize_lambda(medium, wavelength);
        n = lookUp(medium.n, l, 1.0);
    }
    //calculate contribution
    float contrib = TWO_PI * frank_tamm(n, wavelength) * lightParams.trackDist;

    //calculate cherenkov angle
    float cos_theta = 1.0 / n;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    //sample ray direction
    float phi = TWO_PI * random(idx, dim);
    vec3 localDir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    //rotate in particle direction
    vec3 rayDir = createLocalCOSY(lightParams.trackDir) * localDir;

    #ifdef POLARIZATION
    //reference direction is perpendicular to light and particle direction
    vec3 polRef = normalize(cross(rayDir, lightParams.trackDir));
    //light is linear polarized in plane of rayDir and particle dir
    vec4 stokes = vec4(1.0, 1.0, 0.0, 0.0);

    return SourceRay(position, rayDir, stokes, polRef, startTime, contrib);
    #else
    //create and return ray
    return SourceRay(position, rayDir, startTime, contrib);
    #endif
}

SourceRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    uint idx, uint dim
) {
    //get refractive index
    float n = 1.0;
    if (lightParams.medium != uvec2(0)) {
        Medium medium = Medium(lightParams.medium);
        float l = normalize_lambda(medium, wavelength);
        n = lookUp(medium.n, l, 1.0);
    }
    //calculate cherenkov angle
    float cos_theta = 1.0 / n;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));

    //get point on track closest to observer
    float mu = dot(observer - lightParams.trackStart, lightParams.trackDir);
    vec3 C = lightParams.trackStart + mu * lightParams.trackDir;
    float d = distance(observer, C);
    //get point of light source
    mu -= cos_theta / sin_theta * d;
    vec3 position = lightParams.trackStart + mu * lightParams.trackDir;
    vec3 rayDir = normalize(observer - position);
    float u = mu / lightParams.trackDist;
    float startTime = mix(lightParams.startTime, lightParams.endTime, u);

    //calculate contribution
    float contrib = frank_tamm(n, wavelength);
    contrib /= d;
    //set contrib to zero if light source is not on track
    contrib *= float(mu >= 0.0 && mu <= lightParams.trackDist);

    #ifdef POLARIZATION
    //reference direction is perpendicular to light and particle direction
    vec3 polRef = normalize(cross(rayDir, lightParams.trackDir));
    //light is linear polarized in plane of rayDir and particle dir
    vec4 stokes = vec4(1.0, 1.0, 0.0, 0.0);

    return SourceRay(position, rayDir, stokes, polRef, startTime, contrib);
    #else
    //create and return ray
    return SourceRay(position, rayDir, startTime, contrib);
    #endif
}

#endif
