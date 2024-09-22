#ifndef _INCLUDE_WAVELENGTHSOURCE_QUEUE
#define _INCLUDE_WAVELENGTHSOURCE_QUEUE

#ifndef PHOTON_QUEUE_SIZE
#error "PHOTON_QUEUE_SIZE not defined"
#endif

#include "wavelengthsource.common.glsl"

struct WavelengthQueue {
    float wavelength[PHOTON_QUEUE_SIZE];
    float contrib[PHOTON_QUEUE_SIZE];
};

//LOAD/SAVE macros to avoid copying

#define LOAD_PHOTON(PHOTON, QUEUE, IDX) \
WavelengthSample PHOTON = WavelengthSample(\
    QUEUE.wavelength[IDX], QUEUE.contrib[IDX]);

#define SAVE_PHOTON(PHOTON, QUEUE, IDX) \
    QUEUE.wavelength[IDX] = PHOTON.wavelength;\
    QUEUE.contrib[IDX] = PHOTON.contrib;

#endif
