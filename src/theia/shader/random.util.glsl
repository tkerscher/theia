#ifndef _INCLUDE_RANDOM_UTIL
#define _INCLUDE_RANDOM_UTIL

#define ONE_MINUS_EPSILON uintBitsToFloat(0x3F7FFFFFu)
#define EPSILON uintBitsToFloat(0x2F800000u)

//converts the given 32bit unsigned int to a float in the open interval [0,1)
float normalizeUint(uint i) {
    return min(ONE_MINUS_EPSILON, i * EPSILON);

    //alternative version; faster, but less precise
    //interprets the lower 24bit as mantisse
    // return uintBitsToFloat((state & 0x7FFFFF) | 0x3F800000) - 1.0;
}
vec2 normalizeUint(uvec2 v) {
    return vec2(
        normalizeUint(v.x),
        normalizeUint(v.y)
    );
}
vec3 normalizeUint(uvec3 v) {
    return vec3(
        normalizeUint(v.x),
        normalizeUint(v.y),
        normalizeUint(v.z)
    );
}
vec4 normalizeUint(uvec4 v) {
    return vec4(
        normalizeUint(v.x),
        normalizeUint(v.y),
        normalizeUint(v.z),
        normalizeUint(v.w)
    );
}

#endif
