#ifndef _INCLUDE_WAVELENGTHSOURCE_HOST
#define _INCLUDE_WAVELENGTHSOURCE_HOST

#include "wavelengthsource.queue.glsl"

layout(scalar) readonly buffer WavelengthQueueIn {
    WavelengthQueue queue;
} wavelengthQueueIn;

WavelengthSample sampleWavelength(uint idx, uint dim) {
    LOAD_PHOTON(photon, wavelengthQueueIn.queue, idx)
    return photon;
}

#endif
