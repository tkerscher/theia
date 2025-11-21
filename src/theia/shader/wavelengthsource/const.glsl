#ifndef _INCLUDE_WAVELENGTHSOURCE_CONST
#define _INCLUDE_WAVELENGTHSOURCE_CONST

#include "wavelengthsource.common.glsl"

uniform WavelengthParams {
    float lambda;
} wavelengthParams;

WavelengthSample sampleWavelength(uint idx, uint dim) {
    return WavelengthSample(wavelengthParams.lambda, 1.0);
}

#endif
