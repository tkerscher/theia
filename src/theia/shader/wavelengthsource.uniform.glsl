#ifndef _INCLUDE_WAVELENGTHSOURCE_UNIFORM
#define _INCLUDE_WAVELENGTHSOURCE_UNIFORM

#include "wavelengthsource.common.glsl"

layout(scalar) uniform WavelengthParams {
    float lam_min;
    float lam_max;
    float contrib;
} wavelengthParams;

WavelengthSample sampleWavelength(uint idx, uint i) {
    float u = random(idx, i);
    float lam = mix(wavelengthParams.lam_min, wavelengthParams.lam_max, u);
    return WavelengthSample(lam, wavelengthParams.contrib);
}

#endif
