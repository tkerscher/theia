#ifndef _INCLUDE_WAVELENGTHSOURCE_CONST
#define _INCLUDE_WAVELENGTHSOURCE_CONST

uniform WavelengthParams {
    float lambda;
} wavelengthParams;

float sampleWavelength(uint idx, inout uint dim) {
    return wavelengthParams.lambda;
}

float sampleWavelength(out float contrib, uint idx, inout uint dim) {
    contrib = 1.0;
    return wavelengthParams.lambda;
}

#endif
