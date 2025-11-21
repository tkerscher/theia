#ifndef _INCLUDE_RAY_MEDIUM
#define _INCLUDE_RAY_MEDIUM

//thanks to a circular dependency, this util function had to be
//expelled into a separate file

// #include "ray.glsl"

//get medium util function
//returns the medium the ray currently resides in
#ifndef USE_GLOBAL_MEDIUM
uint getMediumIdx(ForwardRay ray) {
    return getMediumIdx(ray.state);
}
uint getMediumIdx(BackwardRay ray) {
    return getMediumIdx(ray.state);
}
#else
//here we expect a getMediumIdx() for fetching the global medium to be defined
//to hide this detail, we still define functions taking a ray.
uint getMediumIdx(ForwardRay ray) {
    return getMediumIdx();
}
uint getMediumIdx(BackwardRay ray) {
    return getMediumIdx();
}
#endif

#endif
