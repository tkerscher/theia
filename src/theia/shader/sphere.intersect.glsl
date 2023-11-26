#ifndef _SPHERE_INTERSECT_INCLUDE
#define _SPHERE_INTERSECT_INCLUDE

#include "sphere.glsl"

//simple ray sphere intersection following
//Chapter 7 in Ray Tracing Gems 2 by Marrs, A. et. al.
//returns the distance along dir to first sphere hit, or +inf if miss
float intersectSphere(const Sphere sphere, vec3 observer, vec3 dir) {
    vec3 f = observer - sphere.position;
    float b = -dot(f, dir);

    //shortcut: if b < 0.0 -> sphere behind ray -> no hit
    if (subgroupAll(b <= 0.0)) {
        return 1.0 / 0.0; //+inf
    }
    else if (b <= 0.0) {
        //mixed branching
        return 1.0 / 0.0; //+inf
    }

    float r2 = sphere.radius * sphere.radius;
    vec3 dd = f + b*dir;
    float d = r2 - dot(dd, dd);

    //let the compiler add checks to see if all invocations use the same branch
    //might dampen the impact of branching (skipping dead branch)
    if (subgroupAll(d < 0.0)) {
        return 1.0 / 0.0; //+inf
    }
    else if (subgroupAll(d >= 0.0)) {
        float c = dot(f, f) - r2;
        float q = b + sign(b) * sqrt(d);
        return c / q;
    }
    else {
        //mixed branching
        if (d < 0.0) {
            return 1.0 / 0.0; //+inf
        }
        else {
            float c = dot(f, f) - r2;
            float q = b + sign(b) * sqrt(d);
            return c / q;
        }
    }    
}

#endif
