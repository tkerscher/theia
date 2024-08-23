#ifndef _INCLUDE_LIGHTSOURCE_HOST
#define _INCLUDE_LIGHTSOURCE_HOST

#include "lightsource.queue.glsl"

//read and return rays from a buffer

layout(scalar) readonly buffer LightQueueIn {
    LightSourceQueue queue;
} lightQueueIn;

SourceRay sampleLight(uint idx, uint dim) {
    LOAD_SAMPLE(ray, lightQueueIn.queue, idx)
    return ray;
}

#endif
