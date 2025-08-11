#ifndef _INCLUDE_WAVELENGTHSOURCE_FUNCTION
#define _INCLUDE_WAVELENGTHSOURCE_FUNCTION

#include "lookup.glsl"

uniform WavelengthParams {
    //using a buffer reference allows us to change it without recreating the pipeline
    Table1D table;
    //constant contrib due to importance sampling
    float contrib;
} wavelengthParams;

WavelengthSample sampleWavelength(uint idx, inout uint i) {
    float lam = lookUp(wavelengthParams.table, random(idx, i), 0.0);
    return WavelengthSample(lam, wavelengthParams.contrib);
}

#endif
