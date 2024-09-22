#ifndef _INCLUDE_WAVELENGTHSOURCE_HOST
#define _INCLUDE_WAVELENGTHSOURCE_HOST

#include "wavelengthsource.queue.glsl"

layout(scalar) readonly buffer PhotonQueueIn {
    WavelengthQueue queue;
} photonQueueIn;

WavelengthSample sampleWavelength(uint idx, uint dim) {
    LOAD_PHOTON(photon, photonQueueIn.queue, idx)
    return photon;
}

#endif
