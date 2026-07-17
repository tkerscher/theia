#ifndef _INCLUDE_WAVELENGTHSOURCE_SPECTRUM
#define _INCLUDE_WAVELENGTHSOURCE_SPECTRUM

uniform WavelengthParams {
    //using a buffer reference allows us to change it without recreating the pipeline
    Table1D spectrumTableAddress;
} wavelengthParams;

float sampleWavelength(uint idx, inout uint dim) {
    return lookUp(wavelengthParams.spectrumTableAddress, random(idx, dim), 400.0);
}

float sampleWavelength(out float contrib, uint idx, inout uint dim) {
    contrib = 1.0;
    return lookUp(wavelengthParams.spectrumTableAddress, random(idx, dim), 400.0);
}

#endif
