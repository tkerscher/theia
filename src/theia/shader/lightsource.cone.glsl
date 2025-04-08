#ifndef _INCLUDE_LIGHTSOURCE_CONE
#define _INCLUDE_LIGHTSOURCE_CONE

#include "math.glsl"

layout(scalar) uniform LightParams {
    vec3 position;
    vec3 direction;
    float cosOpeningAngle;

    float contribFwd;
    float contribBwd;

    float t_min;
    float t_max;

    //always keep polarization info to make things easier on the python side
    vec3 polRef;
    vec4 stokes;
} lightParams;

SourceRay sampleLight(
    float wavelength,
    const MediumConstants medium,
    uint idx, inout uint dim
) {
    //sample cone
    vec2 u = random2D(idx, dim);
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
    float v = random(idx, dim);
    float startTime = mix(lightParams.t_min, lightParams.t_max, v);

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
        startTime,
        lightParams.contribFwd
    );
    #else
    //assemble source ray
    return createSourceRay(
        lightParams.position,
        rayDir,
        startTime,
        lightParams.contribFwd
    );
    #endif
}

SourceRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    const MediumConstants medium,
    uint idx, inout uint dim
) {
    //get direction
    vec3 rayDir = normalize(observer - lightParams.position);    
    //calculate contribution (zero if outside cone)
    float cos_angle = dot(rayDir, lightParams.direction);
    float contrib = lightParams.contribBwd * float(cos_angle > lightParams.cosOpeningAngle);
    contrib *= dw_dA(lightParams.position, observer, normal);
    //sample start time
    float u = random(idx, dim);
    float startTime = mix(lightParams.t_min, lightParams.t_max, u);

    #ifdef LIGHTSOURCE_POLARIZED
    //TODO: Check if edge case rayDir || polRef causes problems
    //      (contrib should be zero)
    vec3 polRef = lightParams.polRef;
    polRef -= dot(polRef, rayDir) * rayDir;
    polRef = normalize(polRef);

    //assemble source ray
    return createSourceRay(
        lightParams.position,
        rayDir,
        lightParams.stokes,
        polRef,
        startTime,
        contrib
    );
    #else
    //assemble source ray
    return createSourceRay(
        lightParams.position,
        rayDir,
        startTime,
        contrib
    );
    #endif
}

#endif
