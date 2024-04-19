#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV
#define _INCLUDE_LIGHTSOURCE_CHERENKOV

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

#include "math.glsl"
#include "material.glsl"

//We either want to track photon count or radiance
//TODO: make charge a variable?
#ifndef FRANK_TAMM_USE_PHOTON_COUNT
/***** Frank-Tamm formula
 *  d^2 E                      1   /            1        \
 * ------- = pi*c^2*e^2*mu_0 ----- | 1 - --------------- |
 * dx dlam                   lam^3 \     beta^2*n(lam)^2 /
 */

// pi * c^2 * e^2 * mu_0 (includes factor for µm)
//note, that there's an extra factor 2pi that cancels:
//uniform phi prob & uniform distribtion of energy in phi
#define FRANK_TAMM_CONST 56.84752175519612 // eV / m*nm

float frank_tamm(float n, float lambda) {
    //for better numerical stability, convert lambda from nm -> µm (closer to one)
    lambda *= 1e-3;
    return FRANK_TAMM_CONST / (lambda*lambda*lambda) * (1.0 - (1.0 / (n*n)));
}

#else //sample photon count
/***** Frank-Tamm formula
 *  d^2 N                1   /            1        \     N
 * ------- = 2pi*alpha ----- | 1 - --------------- |  [ --- ]
 * dx dlam             lam^2 \     beta^2*n(lam)^2 /    m*nm
 *
 * alpha: fine structure constant
 */

// 2pi * alpha (includes factor for µm)
#define FRANK_TAMM_CONST 45.85061844473497

float frank_tamm(float n, float lambda) {
    //for better numerical stability, convert lambda from nm -> µm (closer to one)
    lambda *= 1e-3;
    return FRANK_TAMM_CONST / (lambda*lambda) * (1.0 - (1.0 / (n*n)));
}

#endif
#undef FRANK_TAMM_CONST

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

SourceRay sampleLight(uint idx) {
    //fetch track
    ParticleTrack track = ParticleTrack(trackParams.track);

    //randomly select track segment via first vertex [0,length - 1]
    float u = random(idx, RNG_RAY_SAMPLE_OFFSET) * float(track.trackLength);
    //random() should exclude 1.0, but just to be safe
    uint vertexIdx = min(uint(floor(u)), track.trackLength - 1);
    u = fract(u); //use fractal part for interpolation
    //load start and end vertex
    TrackVertex start = track.vertices[vertexIdx];
    TrackVertex end = track.vertices[vertexIdx + 1];
    //interpolate track position
    vec3 pos = mix(start.pos, end.pos, u);
    float time = mix(start.time, end.time, u);

    //sample photon (wavelength)
    SourceSample photon = sampleSource(idx, 0);
    float wavelength = photon.wavelength;
    float startTime = photon.startTime + time;
    float contrib = photon.contrib;

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
    float phi = TWO_PI * random(idx, RNG_RAY_SAMPLE_OFFSET + 1);
    vec3 dir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    //rotate in particel direction
    vec3 particleDir = normalize(end.pos - start.pos);
    vec3 rayDir = createLocalCOSY(particleDir) * dir;

    //update contribution with sampling prob (2pi included in frank-tamm)
    float segmentLength = distance(start.pos, end.pos);
    contrib *= float(track.trackLength) * segmentLength;

    //use Frank-Thamm to calculate radiance
    //clamp to 0.0 to handle n <= 1.0 cases
    contrib *= max(frank_tamm(n, wavelength), 0.0);

#ifdef POLARIZATION
    //reference direction is perpendicular to light and particle direction
    vec3 polRef = normalize(cross(rayDir, particleDir));
    //light is linear polarized in plane of rayDir and particleDir
    vec4 stokes = vec4(1.0, 1.0, 0.0, 0.0);

    return SourceRay(pos, rayDir, stokes, polRef, wavelength, startTime, contrib);
#else
    //create and return ray
    return SourceRay(pos, rayDir, wavelength, startTime, contrib);
#endif
}

#endif
