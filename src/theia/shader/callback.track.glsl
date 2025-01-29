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

//base event handler
void onEvent(const RayState ray, ResultCode code, uint idx, uint i) {
    //ignore max iter event (it does not progress the track)
    if (code == RESULT_CODE_MAX_ITER) return;
    
    //update track length & code; we'll keep the last update
    //(assume ordered call to onEvent())
    trackBuffer.n[idx] = i;
    trackBuffer.code[idx] = code;

    //check boundary
    if (i >= TRACK_LENGTH) return;

    trackBuffer.x[i][idx] = ray.position.x;
    trackBuffer.y[i][idx] = ray.position.y;
    trackBuffer.z[i][idx] = ray.position.z;
    trackBuffer.t[i][idx] = ray.time;
}

//extended handler
#ifdef TRACK_POLARIZED

//common code to declutter
void onEvent(const RayState ray, ResultCode code, vec4 stokes, vec3 polRef, uint idx, uint i) {
    //ignore max iter event (it does not progress the track)
    if (code == RESULT_CODE_MAX_ITER) return;

    onEvent(ray, code, idx, i);

    //check boundary
    if (i >= TRACK_LENGTH) return;

    trackBuffer.i[i][idx] = stokes.x;
    trackBuffer.q[i][idx] = stokes.y;
    trackBuffer.u[i][idx] = stokes.z;
    trackBuffer.v[i][idx] = stokes.w;
    
    trackBuffer.ref_x[i][idx] = polRef.x;
    trackBuffer.ref_y[i][idx] = polRef.y;
    trackBuffer.ref_z[i][idx] = polRef.z;
}

//declare callback to be detailed
#define DETAILED_CALLBACK 1

#ifdef POLARIZATION

void onEvent(const PolarizedForwardRay ray, ResultCode code, uint idx, uint i) {
    onEvent(ray.state, code, ray.stokes, ray.polRef, idx, i);
}
void onEvent(const PolarizedBackwardRay ray, ResultCode code, uint idx, uint i) {
    //we don not have a stokes vector in backward mode -> set stokes to zero
    onEvent(ray.state, code, vec4(0.0), ray.polRef, idx, i);
}

#else

void onEvent(const UnpolarizedForwardRay ray, ResultCode code, uint idx, uint i) {
    vec4 stokes = vec4(1.0, vec3(0.0)); // unpolarized stokes
    vec3 polRef = vec3(0.0);
    onEvent(ray.state, code, stokes, polRef, idx, i);
}
void onEvent(const UnpolarizedBackwardRay ray, ResultCode code, uint idx, uint i) {
    //we don not have a stokes vector in backward mode -> set stokes to zero
    onEvent(ray.state, code, vec4(0.0), vec3(0.0), idx, i);
}

#endif // #ifdef POLARIZATION

#endif // #ifdef TRACK_POLARIZED

#endif
