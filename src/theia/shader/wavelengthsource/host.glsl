#ifndef _INCLUDE_WAVELENGTHSOURCE_HOST
#define _INCLUDE_WAVELENGTHSOURCE_HOST

#include "util/buffers.glsl"

uniform WavelengthParams {
    uvec2 queueAdr;
    uint queueSize;
} wavelengthParams;

#ifdef WAVELENGTH_SOURCE_EMIT_PARTICLE

float sampleWavelength(uint idx, inout uint dim) {
    FloatBuffer floats = FloatBuffer(wavelengthParams.queueAdr);
    return floats.values[idx];
}

#else

float sampleWavelength(out float contrib, uint idx, inout uint dim) {
    FloatBuffer floats = FloatBuffer(wavelengthParams.queueAdr);
    float lambda = floats.values[idx];
    idx += wavelengthParams.queueSize;
    contrib = floats.values[idx];
    return lambda;
}

#endif

#endif
