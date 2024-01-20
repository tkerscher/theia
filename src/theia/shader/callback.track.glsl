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
} trackBuffer;

void onEvent(const Ray ray, ResultCode code, uint idx, uint i) {
    //update track length & code; we'll keep the last update
    //(assume ordered call to onEvent())
    trackBuffer.n[idx] = i;
    trackBuffer.code[idx] = code;

    trackBuffer.x[i][idx] = ray.position.x;
    trackBuffer.y[i][idx] = ray.position.y;
    trackBuffer.z[i][idx] = ray.position.z;
    // //TODO: should we really ignore other samples?
    trackBuffer.t[i][idx] = ray.samples[0].time;
}

#endif
