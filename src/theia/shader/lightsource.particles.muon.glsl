#ifndef _INCLUDE_LIGHTSOURCE_PARTICLES_MUON
#define _INCLUDE_LIGHTSOURCE_PARTICLES_MUON

#include "lightsource.common.glsl"
#include "lightsource.particles.common.glsl"
#include "material.glsl"
#include "util.sample.glsl"

/**
 * Parameterization of the light yield from a muon track and its secondary
 * particles up to 500MeV as described in the above papers.
*/
uniform MuonTrackParams {
    //geometric properties
    vec3 startPosition;
    float startTime;
    vec3 endPosition;
    float endTime;

    //light yield parameters
    float energyScale;
    //angular distribution parameters
    //see notebooks/track_angular_dist_fit.ipynb
    float a_angular;
    float b_angular;
} track;

SourceRay sampleLight(
    float wavelength,
    const MediumConstants medium,
    uint idx, inout uint dim
) {
    //sample point on track
    float u = random(idx, dim);
    vec3 position = mix(track.startPosition, track.endPosition, u);
    float startTime = mix(track.startTime, track.endTime, u);
    float contrib = distance(track.startPosition, track.endPosition);
    //add secondary particles' light yield by rescaling contrib
    contrib *= track.energyScale;

    //sample emission direction
    vec2 v = random2D(idx, dim); //2D for stratification
    float phi = TWO_PI * v.x;
    float cos_theta = particle_sampleEmissionAngle(
        medium.n, track.a_angular, track.b_angular, v.y);
    //assemble ray direction
    vec3 trackDir = normalize(track.endPosition - track.startPosition);
    vec3 rayDir = createLocalCOSY(trackDir) * sphericalToCartessian(phi, cos_theta);

    //if we importance sample the Frank-Tamm formula, we must not apply it here
    //we assume the constant factor was applied elsewhere (e.g. wavelength source)
    #ifndef FRANK_TAMM_IS
    contrib *= frank_tamm(wavelength, medium.n);
    #endif

    //return source ray
    return createSourceRay(position, rayDir, startTime, contrib);
    //NOTE: For now we don't have polarization here
}

SourceRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    const MediumConstants medium,
    uint idx, inout uint dim
) {
    //as we expect muon tracks to be rather long, we want to importance sample
    //the attenuation factor exp(-mu_t*t). Let C be the point on the line through
    //the track with minimal distance to observer and x the signed distance
    //between the sampled point and C. We can than write t^2 = d^2 + x^2
    //where d is the minimal distance between C and the observer. Positive
    //x mean we follow the track direction and vice versa.
    //
    //Unfortunately, as it turns out, the above function cannot be annalytically
    //integrated and thus we have to use a proxy. We will use 1/(d^2+(ax)^2),
    //whose integral is tan^-1(ax/d)/(ad). Heuristically, choosing
    //a^2 = 2.2 / (b(b+2)); b = ln(2) / (mu_t * d) seems to give a good result.
    //Mathematically, the value of a does not matter and only effects simulation
    //performance.

    float trackDist = distance(track.endPosition, track.startPosition);
    vec3 trackDir = normalize(track.endPosition - track.startPosition);
    //calculate point of minimal distance
    float startDist = distance(track.startPosition, observer);
    float cos_start = dot(normalize(observer - track.startPosition), trackDir);
    //signed distances: <0 -> point lies before C w.r.t. track direction
    float distStart2C = -cos_start * startDist;
    float distEnd2C = trackDist + distStart2C;
    float d = sqrt(max(1.0 - cos_start*cos_start, 0.0)) * startDist; //dist(observer, C)

    //calculate shape parameter
    float b = log(2) / (d * medium.mu_e);
    float a2 = 2.2 / (b * (b + 2.0));
    float a = sqrt(a2);
    //normalize proxy pdf
    //Note, that we miss a factor a*d here, which cancels in the inverse CDF
    float int_lo = atan(a*distStart2C/d);
    float int_hi = atan(a*distEnd2C/d);
    float norm = int_hi - int_lo; //factor 1/(a*d) missing on purpose
    //sample x
    float u = random(idx, dim);
    u = u * norm + int_lo;
    float x = d / a * tan(u); 
    //calculate contrib ~ 1/p(x)
    float contrib = norm / (a * d) * (d*d + a2*x*x);

    //assemble ray
    x -= distStart2C; //shift so x=0 -> rayPos = startPos
    vec3 rayPos = track.startPosition + x * trackDir;
    vec3 rayDir = normalize(observer - rayPos);
    float time = track.startTime + x * INV_SPEED_OF_LIGHT;
    //convert integral dA -> dw
    contrib *= dw_dA(rayPos, observer, normal);

    //evaluate angular emission profile
    float cos_obs = dot(trackDir, rayDir);
    contrib *= particle_evalEmissionAngle(
        medium.n, track.a_angular, track.b_angular, cos_obs
    );

    //apply energy scale and Frank-Tamm if needed
    contrib *= track.energyScale;
    #ifndef FRANK_TAMM_IS
    contrib *= frank_tamm(wavelength, medium.n);
    #endif
    //1/2pi stems from the missing d/d(phi) in the frank tamm formula
    contrib *= INV_2PI;

    //assemble ray and return
    return createSourceRay(rayPos, rayDir, time, contrib);
}

#endif
