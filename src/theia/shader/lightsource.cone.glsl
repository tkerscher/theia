#ifndef _INCLUDE_LIGHTSOURCE_CONE
#define _INCLUDE_LIGHTSOURCE_CONE

#include "math.glsl"

layout(scalar) uniform LightParams {
    vec3 position;
    vec3 direction;
    float cosOpeningAngle;
    float contrib;

    float t_min;
    float t_max;

    //always keep polarization info to make things easier on the python side
    vec3 polRef;
    vec4 stokes;
} lightParams;

SourceRay sampleLight(uint idx, uint dim) {
    //sample cone
    vec2 u = random2D(idx, dim); dim += 2;
    float phi = TWO_PI * u.x;
    float cos_theta = (1.0 - u.y) + lightParams.cosOpeningAngle * u.y;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    //construct local ray dir
    vec3 localDir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    //convert to global space
    mat3 trafo = createLocalCOSY(normalize(lightParams.direction));
    vec3 rayDir = trafo * localDir;

    //sample startTime
    float v = random(idx, dim); dim++;
    float startTime = mix(lightParams.t_min, lightParams.t_max, v);

    //sample photon
    WavelengthSample photon = sampleWavelength(idx, dim);
    //apply budget
    photon.contrib *= lightParams.contrib;

    #ifdef LIGHTSOURCE_POLARIZED
    //make polRef orthogonal to light ray direction
    //Obviously fails for rayDir || polRef
    // -> Check this on the python side
    vec3 polRef = lightParams.polRef;
    polRef -= dot(polRef, rayDir) * rayDir;
    polRef = normalize(polRef);

    //assemble source ray
    return createSourceRay(
        lightParams.position,
        rayDir,
        startTime,
        // lightParams.stokes,
        vec4(1.0, 0.9, 0.1, -0.5),
        polRef,
        photon
    );
    #else
    //assemble source ray
    return createSourceRay(
        lightParams.position,
        rayDir,
        startTime,
        photon
    );
    #endif
}

#endif
