#ifndef _INCLUDE_CALLBACK_TRACK
#define _INCLUDE_CALLBACK_TRACK

writeonly buffer TrackBuffer {
    uint n[TRACK_COUNT];

    float x[TRACK_LENGTH][TRACK_COUNT];
    float y[TRACK_LENGTH][TRACK_COUNT];
    float z[TRACK_LENGTH][TRACK_COUNT];
    float t[TRACK_LENGTH][TRACK_COUNT];

    ResultCode codes[TRACK_LENGTH][TRACK_COUNT];
} trackBuffer;

void saveTrackPoint(vec3 pos, float t, ResultCode code, uint idx, uint i) {
    //always save path length to know later if we ran out of memory
    trackBuffer.n[idx] = i;

    if (i >= TRACK_LENGTH) return;
    trackBuffer.x[i][idx] = pos.x;
    trackBuffer.y[i][idx] = pos.y;
    trackBuffer.z[i][idx] = pos.z;
    trackBuffer.t[i][idx] = t;

    trackBuffer.codes[i][idx] = code;
}

#ifdef ForwardRay

void onEvent(const ForwardRay ray, ResultCode code, uint idx, uint i) {
    #ifdef RAY_TRANSIENT
    saveTrackPoint(ray.position, ray.time, code, idx, i);
    #else
    saveTrackPoint(ray.position, 0.0, code, idx, i);
    #endif
}

#endif

#ifdef BackwardRay

void onEvent(const BackwardRay ray, ResultCode code, uint idx, uint i) {
    #ifdef RAY_TRANSIENT
    saveTrackPoint(ray.position, ray.time, code, idx, i);
    #else
    saveTrackPoint(ray.position, 0.0, code, idx, i);
    #endif
}

#endif

#endif
