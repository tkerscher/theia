#ifndef _INCLUDE_WAVELENGTHSOURCE_UNIFORM
#define _INCLUDE_WAVELENGTHSOURCE_UNIFORM

uniform WavelengthParams {
    float lam_min;
    float lam_max;
    float contrib;
} wavelengthParams;

float sampleWavelength(uint idx, inout uint dim) {
    return mix(
        wavelengthParams.lam_min,
        wavelengthParams.lam_max,
        random(idx, dim)
    );
}

float sampleWavelength(out float contrib, uint idx, inout uint dim) {
    contrib = wavelengthParams.contrib;
    return sampleWavelength(idx, dim);
}

#endif
