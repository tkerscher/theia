#ifndef _INCLUDE_VOLUME_TRACER_SHADOWING
#define _INCLUDE_VOLUME_TRACER_SHADOWING

#ifndef DISABLE_SELF_SHADOWING

#include "target/common.glsl"
#include "target.glsl"

bool isVisible(vec3 observer, vec3 target) {
    vec3 dir = normalize(target - observer);
    float dist = distance(target, observer);

    //Check if we are shadowed by target
    // -> returns false (target not visible) if shadowed
    TargetSample hit = intersectTarget(observer, dir);
    return !hit.valid || (hit.dist >= dist);
}

#else

bool isVisible(vec3 observer, vec3 target) {
    return true;
}

#endif

#endif
