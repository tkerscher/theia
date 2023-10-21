#ifndef _ITEMS_INCLUDE
#define _ITEMS_INCLUDE

#include "ray.glsl"

struct RayItem{
    Ray ray;
    int targetIdx; //same as customIdx on geometry
};

struct PhotonHit{
    float wavelength;
    float travelTime;
    float log_radiance;
    float throughput;
};

PhotonHit createHit(Photon photon) {
    //we just have to combine the throughputs and transform the data
    float throughput = exp(photon.T_log) * photon.T_lin;
    return PhotonHit(
        photon.wavelength,
        photon.travelTime,
        photon.log_radiance,
        throughput
    );
}

struct ResponseItem{
    //in object space (no trafo)
    vec3 position;
    vec3 direction;
    vec3 normal;
    int detectorIdx;

    PhotonHit hits[N_PHOTONS];
};

struct ShadowRayItem{
    Ray ray;
    int targetIdx;

    float dist;
};

struct IntersectionItem{
    Ray ray;
    int targetIdx;

    int geometryIdx;
    int customIdx;
    int triangleIdx;
    vec2 barys;
    mat4x3 obj2World;
    mat4x3 world2Obj;
};

struct VolumeScatterItem{
    Ray ray;
    int targetIdx;

    float dist;
};

#endif
