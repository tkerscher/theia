#ifndef _INCLUDE_RAY_SAMPLE
#define _INCLUDE_RAY_SAMPLE

#include "lightsource.common.glsl"
#include "material.glsl"
#include "ray.glsl"

//converts a source into a tracing ray
Ray createRay(const SourceRay source, const Medium medium) {
    return Ray(
        source.position,
        normalize(source.direction), //just to be safe
#ifndef USE_GLOBAL_MEDIUM
        uvec2(medium),
#endif
#ifdef POLARIZATION
        source.stokes,
        source.polRef,
#endif
        source.wavelength,
        source.startTime,
        source.contrib, //lin_contrib
        0.0,            //log_contrib
        lookUpMedium(medium, source.wavelength)
    );
}

#endif
