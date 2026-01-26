#ifndef _INCLUDE_UTIL_JACOBIAN
#define _INCLUDE_UTIL_JACOBIAN

//jacobian transforming an integral over area (dA) to one over solid angles (dw)
//normal is the surface normal at the target point or zero if a volume point:
//       |cos|
// dw = ------- * dA
//        r^2
float dw_dA(vec3 observer, vec3 target, vec3 normal) {
    vec3 dir = target - observer;
    float r2 = dot(dir, dir);
    dir = normalize(dir);
    
    float cos_nrm = (normal == vec3(0.0)) ? 1.0 : abs(dot(dir, normal));
    return cos_nrm / r2;
}

//jacobian transforming an integral over solid angled (dw) to one over area (dA)
//normal is the surface normal at the target point or zero if a volume point
float dA_dw(vec3 observer, vec3 target, vec3 normal) {
    vec3 dir = target - observer;
    float d2 = dot(dir, dir);
    dir = normalize(dir);
    
    float cos_nrm = (normal == vec3(0.0)) ? 1.0 : abs(dot(dir, normal));
    float factor = d2 / cos_nrm;
    //for dot(dir, nrm) near zero we might get inf as factor
    //-> mark as invalid (return zero)
    return isinf(factor) ? 0.0 : factor;
}

#endif
