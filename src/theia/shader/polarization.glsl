#ifndef _INCLUDE_POLARIZATION
#define _INCLUDE_POLARIZATION

#include "material.glsl"

mat4 rotatePolRef(float phi) {
    float c = cos(2.0*phi);
    float s = sin(2.0*phi);
    //assemble reference frame rotation matrix (column major!)
    return mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0,   c,   s, 0.0,
        0.0,  -s,   c, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

//looks up the phase matrix for a 
mat4 lookUpPhaseMatrix(const Medium medium, float cos_theta) {
    if (uint64_t(medium) == 0) {
        return mat4(1.0);
    }

    //look up matrix elements
    float t = 0.5 * (cos_theta + 1.0);
    float m12 = lookUp(medium.phase_m12, t, 0.0);
    float m22 = lookUp(medium.phase_m22, t, 0.0);
    float m33 = lookUp(medium.phase_m33, t, 0.0);
    float m34 = lookUp(medium.phase_m34, t, 0.0);

    //assemble matrix (column major!)
    return mat4(
        1.0, m12, 0.0, 0.0,
        m12, m22, 0.0, 0.0,
        0.0, 0.0, m33, -m34,
        0.0, 0.0, m34, m33
    );
}

mat4 polarizerMatrix(float attenuation, float p, float s) {
    float m12 = (p*p - s*s) / attenuation;
    float m33 = (2.0*p*s) / attenuation;
    //assemble matrix (column major!)
    return mat4(
        1.0, m12, 0.0, 0.0,
        m12, 1.0, 0.0, 0.0,
        0.0, 0.0, m33, 0.0,
        0.0, 0.0, 0.0, m33
    );
}

#endif
