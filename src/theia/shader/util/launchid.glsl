#ifndef _INCLUDE_UTIL_LAUNCH_ID
#define _INCLUDE_UTIL_LAUNCH_ID

#include "util/hash.glsl"

/**
 * Function returning either the invocation or launch id depending on the
 * current shader stage.
*/
uint getLaunchId() {
    #ifdef RAY_TRACING_PIPELINE
    return gl_LaunchIDEXT.x;
    #else
    return gl_GlobalInvocationID.x;
    #endif
}

/**
 * Returns a scrambled launch id by hashing it.
*/
uint getScrambledLaunchId(uint stride) {
    const uint seed = 0xC34DD9A5u;
    return hash(getLaunchId(), seed) % stride;
}

#endif
