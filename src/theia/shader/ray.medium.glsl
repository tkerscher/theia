#ifndef _INCLUDE_RAY_MEDIUM
#define _INCLUDE_RAY_MEDIUM

//thanks to a circular dependency, this util function had to be
//expelled into a separate file

// #include "ray.glsl"

//get medium util function
//returns the medium the ray currently resides in
#ifndef USE_GLOBAL_MEDIUM
Medium getMedium(ForwardRay ray) {
    return getMedium(ray.state);
}
Medium getMedium(BackwardRay ray) {
    return getMedium(ray.state);
}
#else
//here we expect a getMedium() for fetching the global medium to be defined
//to hide this detail, we still define funtions taking a ray.
Medium getMedium(ForwardRay ray) {
    return getMedium();
}
Medium getMedium(BackwardRay ray) {
    return getMedium();
}
#endif

#endif
