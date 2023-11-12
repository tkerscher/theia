#ifndef _ITEMS_INCLUDE
#define _ITEMS_INCLUDE

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 32
#endif

#include "ray.glsl"

struct ShadowRayItem{
    Ray ray;            //40 + N*32 bytes

    float dist;         // 4 bytes
};              // TOTAL: 44 + N*32 bytes

struct IntersectionItem{
    Ray ray;            //40 + N*32 bytes

    int geometryIdx;    // 4 bytes
    int customIdx;      // 4 bytes
    int triangleIdx;    // 4 bytes
    vec2 barys;         // 8 bytes
    mat4x3 obj2World;   //12 bytes
    mat4x3 world2Obj;   //12 bytes
};              // TOTAL: 84 + N*32 bytes

struct VolumeScatterItem{
    Ray ray;            //40 + N*32 bytes

    float dist;         // 4 bytes
};              // TOTAL: 44 + N*32 bytes

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
