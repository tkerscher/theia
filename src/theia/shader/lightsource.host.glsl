#ifndef _INCLUDE_LIGHTSOURCE_HOST
#define _INCLUDE_LIGHTSOURCE_HOST

#include "lightsource.queue.glsl"

//read and return rays from a buffer

readonly buffer LightQueueIn {
    LightSourceQueue queue;
} lightQueueIn;

SourceRay sampleLight(
    float wavelength,
    const MediumConstants medium,    
    uint idx, uint dim
) {
    LOAD_SAMPLE(ray, lightQueueIn.queue, idx)
    return ray;
}

#endif
