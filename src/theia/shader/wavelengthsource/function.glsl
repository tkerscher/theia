#ifndef _INCLUDE_WAVELENGTHSOURCE_FUNCTION
#define _INCLUDE_WAVELENGTHSOURCE_FUNCTION

#include "lookup.glsl"

uniform WavelengthParams {
    //using a buffer reference allows us to change it without recreating the pipeline
    Table1D table;
    //constant contrib due to importance sampling
    float contrib;
} wavelengthParams;

float sampleWavelength(uint idx, inout uint dim) {
    return lookUp(wavelengthParams.table, random(idx, dim), 0.0);
}

float sampleWavelength(out float contrib, uint idx, inout uint dim) {
    contrib = wavelengthParams.contrib;
    return lookUp(wavelengthParams.table, random(idx, dim), 0.0);
}

#endif
