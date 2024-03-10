#ifndef _INCLUDE_RAY_SAMPLE
#define _INCLUDE_RAY_SAMPLE

#include "lightsource.common.glsl"
#include "material.glsl"
#include "ray.glsl"

//converts a source into a tracing ray
Ray createRay(const SourceRay source, const Medium medium) {
    Sample samples[N_LAMBDA];
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        samples[i] = Sample(
            source.samples[i].wavelength,
            source.samples[i].startTime,
            source.samples[i].contrib,
            0.0,
            lookUpMedium(medium, source.samples[i].wavelength));
    }
    return Ray(
        source.position,
        normalize(source.direction), //just to be safe
#ifndef USE_GLOBAL_MEDIUM
        uvec2(medium),
#endif
        samples
    );
}

#endif
