#ifndef _INCLUDE_LIGHTSOURCE_CONE
#define _INCLUDE_LIGHTSOURCE_CONE

#include "math.glsl"
#include "util/jacobian.glsl"

uniform LightParams {
    vec3 direction;
    float cosOpeningAngle;
    vec3 position;

    uint mediumIdx;

    float contribFwd;
    float contribBwd;

    float t_min;
    float t_max;

} lightParams;

ForwardRay sampleLight(uint idx, inout uint dim) {
    //sample wavelength using wavelength source
    #ifdef LIGHT_SOURCE_EMIT_PARTICLE
    float wavelength = sampleWavelength(idx, dim);
    #else
    float contrib;
    float wavelength = sampleWavelength(contrib, idx, dim);
    #endif

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

    //assemble forward ray
    return createForwardRay(
        lightParams.position,
        rayDir,
        wavelength,
        lightParams.mediumIdx,
        startTime
        #ifndef LIGHT_SOURCE_EMIT_PARTICLE
        , contrib * lightParams.contribFwd
        #endif
    );
}

#ifndef LIGHT_SOURCE_EMIT_PARTICLE

ForwardRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    uint mediumIdx,
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

    //assemble forward ray
    return createForwardRay(
        lightParams.position,
        rayDir,
        wavelength,
        lightParams.mediumIdx,
        startTime,
        contrib
    );
}

#endif

#endif
