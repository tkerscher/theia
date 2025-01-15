#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV
#define _INCLUDE_LIGHTSOURCE_CHERENKOV

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

#include "math.glsl"
#include "material.glsl"

#include "lightsource.cherenkov.common.glsl"

struct TrackVertex {
    vec3 pos;
    float time;
};

layout(buffer_reference, scalar, buffer_reference_align=4) buffer ParticleTrack {
    uint trackLength;
    TrackVertex vertices[];
};

layout(scalar) uniform TrackParams {
    uvec2 medium;
    uvec2 track;
} trackParams;

SourceRay sampleLight(float wavelength, uint idx, inout uint dim) {
    //fetch track
    ParticleTrack track = ParticleTrack(trackParams.track);

    //randomly select track segment via first vertex [0,length - 1]
    float u = random(idx, dim);
    u *= float(track.trackLength);
    //random() should exclude 1.0, but just to be safe
    uint vertexIdx = min(uint(floor(u)), track.trackLength - 1);
    u = fract(u); //use fractal part for interpolation
    //load start and end vertex
    TrackVertex start = track.vertices[vertexIdx];
    TrackVertex end = track.vertices[vertexIdx + 1];
    //interpolate track position
    vec3 pos = mix(start.pos, end.pos, u);
    float time = mix(start.time, end.time, u);

    //get refractive index
    float n = 1.0;
    if (trackParams.medium != uvec2(0)) {
        Medium medium = Medium(trackParams.medium);
        float l = normalize_lambda(medium, wavelength);
        n = lookUp(medium.n, l, 1.0);
    }    

    //calculate cherenkov angle (assume beta = 1.0)
    float cos_theta = 1.0 / n;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    //sample ray direction
    float phi = TWO_PI * random(idx, dim);
    vec3 dir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    //rotate in particel direction
    vec3 particleDir = normalize(end.pos - start.pos);
    vec3 rayDir = createLocalCOSY(particleDir) * dir;

    //update contribution with sampling prob
    float segmentLength = distance(start.pos, end.pos);
    //calculate contribution
    float contrib = TWO_PI * float(track.trackLength) * segmentLength;
    //use Frank-Tamm to calculate radiance
    contrib *= frank_tamm(n, wavelength);

#ifdef POLARIZATION
    //reference direction is perpendicular to light and particle direction
    vec3 polRef = normalize(cross(rayDir, particleDir));
    //light is linear polarized in plane of rayDir and particleDir
    vec4 stokes = vec4(1.0, 1.0, 0.0, 0.0);

    return SourceRay(pos, rayDir, stokes, polRef, time, contrib);
#else
    //create and return ray
    return SourceRay(pos, rayDir, time, contrib);
#endif
}

//TODO: Implement light sampling for backward mode
//      See simple cherenkov light source and weighted reservoir sampling

#endif
