#ifndef _ITEMS_INCLUDE
#define _ITEMS_INCLUDE

//Check missing macro settings
#ifndef LOCAL_SIZE
#error "missing macro setting: LOCAL_SIZE"
#endif

#include "ray.glsl"

struct IntersectionQueue {
    //40 + 32*N_PHOTONS bytes
    RayQueue rays;

    //116 bytes
    int geometryIdx[QUEUE_SIZE];
    int customIdx[QUEUE_SIZE];
    int triangleIdx[QUEUE_SIZE];
    float baryU[QUEUE_SIZE];
    float baryV[QUEUE_SIZE];
    //mat4x3 obj2World matrix ij -> i-th column, j-th row
    float obj2World00[QUEUE_SIZE];
    float obj2World01[QUEUE_SIZE];
    float obj2World02[QUEUE_SIZE];
    float obj2World10[QUEUE_SIZE];
    float obj2World11[QUEUE_SIZE];
    float obj2World12[QUEUE_SIZE];
    float obj2World20[QUEUE_SIZE];
    float obj2World21[QUEUE_SIZE];
    float obj2World22[QUEUE_SIZE];
    float obj2World30[QUEUE_SIZE];
    float obj2World31[QUEUE_SIZE];
    float obj2World32[QUEUE_SIZE];
    //mat4x3 world2Obj matrix ij -> i-th column, j-th row
    float world2Obj00[QUEUE_SIZE];
    float world2Obj01[QUEUE_SIZE];
    float world2Obj02[QUEUE_SIZE];
    float world2Obj10[QUEUE_SIZE];
    float world2Obj11[QUEUE_SIZE];
    float world2Obj12[QUEUE_SIZE];
    float world2Obj20[QUEUE_SIZE];
    float world2Obj21[QUEUE_SIZE];
    float world2Obj22[QUEUE_SIZE];
    float world2Obj30[QUEUE_SIZE];
    float world2Obj31[QUEUE_SIZE];
    float world2Obj32[QUEUE_SIZE];
};  // TOTAL 156 + 32*N_PHOTONS bytes

struct ShadowRayQueue {
    //40 + 32*N_PHOTONS bytes
    RayQueue rays;
    float dist[QUEUE_SIZE];
};  //TOTAL: 44 + 32*N_PHOTONS bytes

struct VolumeScatterQueue {
    //40 + 32*N_PHOTONS bytes
    RayQueue rays;
    float dist[QUEUE_SIZE];
};  //TOTAL: 44 + 32*N_PHOTONS bytes

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
