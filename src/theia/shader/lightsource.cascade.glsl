#ifndef _INCLUDE_LIGHTSOURCE_CASCADE
#define _INCLUDE_LIGHTSOURCE_CASCADE

#include "lightsource.particles.cascade.glsl"

layout(scalar) uniform LightSourceParams {
    Cascade cascade;
} lightSourceParams;

SourceRay sampleLightSource(
    float wavelength,
    const MediumConstants medium,
    uint idx, inout uint dim
) {
    return sampleCascade(
        lightSourceParams.cascade,
        wavelength, medium,
        idx, dim
    );
}

SourceRay sampleLightSource(
    vec3 observer, vec3 normal,
    float wavelength,
    const MediumConstants medium,
    uint idx, inout uint dim
) {
    return sampleCascade(
        lightSourceParams.cascade,
        observer, normal,
        wavelength, medium,
        idx, dim
    );
}

#endif
