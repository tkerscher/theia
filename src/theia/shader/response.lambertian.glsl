#ifndef _INCLUDE_RESPONSE_LAMBERT
#define _INCLUDE_RESPONSE_LAMBERT

//simple lambert cosine law

float response(
    vec3 rayPos,
    vec3 rayDir,
    vec3 detNormal,
    float wavelength,
    float log_radiance,
    uint detId
) {
    return -dot(rayDir, detNormal) * log_radiance;
}

#endif
