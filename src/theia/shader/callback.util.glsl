#ifndef _INCLUDE_CALLBACK_UTIL
#define _INCLUDE_CALLBACK_UTIL

//Callback can either be minmal or detailed depending on whether they are only
//interested in the ray's state or the complete (i.e. optional polarization)
//To not clutter the code, we always pretend to have a detailed callback and
//provide suitable wrapper if needed.

#ifndef DETAILED_CALLBACK

void onEvent(const ForwardRay ray, ResultCode code, uint idx, uint i) {
    onEvent(ray.state, code, idx, i);
}
void onEvent(const BackwardRay ray, ResultCode code, uint idx, uint i) {
    onEvent(ray.state, code, idx, i);
}

#endif

#endif
