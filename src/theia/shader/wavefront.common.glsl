#ifndef _ITEMS_INCLUDE
#define _ITEMS_INCLUDE

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 32
#endif

#include "ray.glsl"

struct ShadowRayItem{
    Ray ray;

    float dist;
};

struct IntersectionItem{
    Ray ray;

    int geometryIdx;
    int customIdx;
    int triangleIdx;
    vec2 barys;
    mat4x3 obj2World;
    mat4x3 world2Obj;
};

struct VolumeScatterItem{
    Ray ray;

    float dist;
};

struct TraceParams {
    //Index of target we want to hit
    uint targetIdx;

    //for transient rendering, we won't importance sample the media
    float scatterCoefficient;

    //boundary conditions
    float maxTime;
    vec3 lowerBBoxCorner;
    vec3 upperBBoxCorner;
};

#endif
