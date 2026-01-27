#ifndef _INCLUDE_UTIL_OFFSET
#define _INCLUDE_UTIL_OFFSET

/**
 * Offsets ray position from surface hits to prevent self-intersection, i.e.
 * this ensures that after transmission/reflection the ray is actually on the
 * correct side of the geometry by compensating for the finite numerical
 * precision while introducing minimal bias.
 *
 * Normal (n) points outwards for rays existing the surface, else is flipped.
 * Offsets in normal direction.
 *
 * Taken from Ray Tracing Gems: Chapter 6
 * C. Waechter and N. Binder (2019): "A Fast and Robust Method for Avoiding
 * Self-Intersection"
*/
vec3 offsetRay(vec3 p, vec3 n) {
    ivec3 of_i = ivec3(256.0 * n);
    
    vec3 p_i = vec3(
        intBitsToFloat(floatBitsToInt(p.x)+((p.x < 0.0) ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(p.y)+((p.y < 0.0) ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(p.z)+((p.z < 0.0) ? -of_i.z : of_i.z))
    );

    return vec3(
        abs(p.x) < (1.0 / 32.0) ? p.x+ (1.0/65536.0)*n.x : p_i.x,
        abs(p.y) < (1.0 / 32.0) ? p.y+ (1.0/65536.0)*n.y : p_i.y,
        abs(p.z) < (1.0 / 32.0) ? p.z+ (1.0/65536.0)*n.z : p_i.z
    );
}

#endif
