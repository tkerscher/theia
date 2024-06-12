#ifndef _INCLUDE_LIGHTSOURCE_CONE
#define _INCLUDE_LIGHTSOURCE_CONE

#include "math.glsl"

layout(scalar) uniform LightParams {
    vec3 position;
    vec3 direction;
    float cosOpeningAngle;
    float budget;

    //always keep polarization info to make things easier on the python side
    vec3 polRef;
    vec4 stokes;
} lightParams;

SourceRay sampleLight(uint idx) {
    //sample cone
    vec2 u = random2D(idx, 0);
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

    //sample photon
    SourceSample photon = sampleSource(idx, 2);
    //apply budget
    photon.contrib *= lightParams.budget;

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
        lightParams.stokes,
        polRef,
        photon
    );
    #else
    //assemble source ray
    return createSourceRay(
        lightParams.position,
        rayDir,
        photon
    );
    #endif
}

#endif
