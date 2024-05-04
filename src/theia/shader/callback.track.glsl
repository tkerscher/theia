#ifndef _INCLUDE_CALLBACK_TRACK
#define _INCLUDE_CALLBACK_TRACK

#include "ray.glsl"

layout(scalar) writeonly buffer TrackBuffer {
    uint n[TRACK_COUNT];
    ResultCode code[TRACK_COUNT];

    float x[TRACK_LENGTH][TRACK_COUNT];
    float y[TRACK_LENGTH][TRACK_COUNT];
    float z[TRACK_LENGTH][TRACK_COUNT];
    float t[TRACK_LENGTH][TRACK_COUNT];

    #ifdef TRACK_POLARIZED
    float i[TRACK_LENGTH][TRACK_COUNT];
    float q[TRACK_LENGTH][TRACK_COUNT];
    float u[TRACK_LENGTH][TRACK_COUNT];
    float v[TRACK_LENGTH][TRACK_COUNT];

    float ref_x[TRACK_LENGTH][TRACK_COUNT];
    float ref_y[TRACK_LENGTH][TRACK_COUNT];
    float ref_z[TRACK_LENGTH][TRACK_COUNT];
    #endif
} trackBuffer;

void onEvent(const Ray ray, ResultCode code, uint idx, uint i) {
    //update track length & code; we'll keep the last update
    //(assume ordered call to onEvent())
    trackBuffer.n[idx] = i;
    trackBuffer.code[idx] = code;

    trackBuffer.x[i][idx] = ray.position.x;
    trackBuffer.y[i][idx] = ray.position.y;
    trackBuffer.z[i][idx] = ray.position.z;
    trackBuffer.t[i][idx] = ray.time;

    #ifdef TRACK_POLARIZED
    //check if ray has polarization state
    #ifdef POLARIZATION
    trackBuffer.i[i][idx] = ray.stokes.x;
    trackBuffer.q[i][idx] = ray.stokes.y;
    trackBuffer.u[i][idx] = ray.stokes.z;
    trackBuffer.v[i][idx] = ray.stokes.w;
    
    trackBuffer.ref_x[i][idx] = ray.polRef.x;
    trackBuffer.ref_y[i][idx] = ray.polRef.y;
    trackBuffer.ref_z[i][idx] = ray.polRef.z;
    #else
    trackBuffer.i[i][idx] = 1.0;
    trackBuffer.q[i][idx] = 0.0;
    trackBuffer.u[i][idx] = 0.0;
    trackBuffer.v[i][idx] = 0.0;
    
    trackBuffer.ref_x[i][idx] = 0.0;
    trackBuffer.ref_y[i][idx] = 0.0;
    trackBuffer.ref_z[i][idx] = 0.0;
    #endif
    #endif
}

#endif
