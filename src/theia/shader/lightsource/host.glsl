#ifndef _INCLUDE_LIGHTSOURCE_HOST
#define _INCLUDE_LIGHTSOURCE_HOST

uniform LightParams {
    uvec2 queueAdr;
    uint queueSize;
} lightParams;

ForwardRay sampleLight(uint idx, inout uint dim) {
    return loadForwardRay(lightParams.queueAdr, lightParams.queueSize, idx);
}

#endif
