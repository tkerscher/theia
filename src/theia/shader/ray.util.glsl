#ifndef _INCLUDE_RAY_UTIL
#define _INCLUDE_RAY_UTIL

//checks whether ray contains any nan or inf
bool isRayBad(const RayState ray) {
    return 
        any(isnan(ray.position)) ||
        any(isinf(ray.position)) ||
        any(isnan(ray.direction)) ||
        any(isinf(ray.direction)) ||
        length(ray.direction) <= 0.0;
}

bool isRayBad(const ForwardRay ray) {
    return isRayBad(ray.state);
}
bool isRayBad(const BackwardRay ray) {
    return isRayBad(ray.state);
}

#endif