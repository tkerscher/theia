#ifndef _INCLUDE_LIGHTSOURCE_HOST
#define _INCLUDE_LIGHTSOURCE_HOST

//read and return rays from a buffer

layout(scalar) readonly buffer Rays{
    SourceRay rays[];
};

SourceRay sampleLight() {
    uint idx = gl_GlobalInvocationID.x;
    return rays[idx];
}

#endif
