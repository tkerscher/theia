#ifndef _INCLUDE_MATH_SPHERE
#define _INCLUDE_MATH_SPHERE

struct Sphere {
    vec3 position;
    float radius;
};

/**
 * Intersects the given sphere with a provided ray.
 * Returns both the distance to the nearest and farthest intersection point in
 * that order or +inf if there is no intersection.
*/
vec2 intersectSphere(const Sphere sphere, vec3 origin, vec3 direction) {
    //simple ray sphere intersection
    //see Chapter 7 in "Ray Tracing Gems" by E. Haines et. al.
    vec3 f = origin - sphere.position;
    float b2 = dot(f, direction);
    float r2 = sphere.radius * sphere.radius;
    vec3 fd = f - b2 * direction;
    float discr = r2 - dot(fd, fd);

    //default to (+inf, +inf)
    vec2 result = vec2(1.0 / 0.0);
    //sqrt of negative numbers are undefined (by spec)
    // -> need to check
    if (discr >= 0.0) {
        float c = dot(f, f) - r2;
        float q = -b2 - signBit(b2) * sqrt(discr);
        result = vec2(c / q, q);
    }

    //done
    return result;
}

#endif
