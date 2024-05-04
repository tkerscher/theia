#ifndef _INCLUDE_POLARIZATION
#define _INCLUDE_POLARIZATION

#include "material.glsl"
#include "math.glsl"

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

//creates rotation matrix to rotate the ref vector to be ortogonal to the plane
//of scattering from old to new. Assumes all vectors to be normalized
mat4 rotatePolRef(vec3 dir, vec3 ref, vec3 new, out vec3 new_ref) {
    //new reference should be normal to dir and new -> cross product
    //but we have an edge case: if dir == new, cross generates zero
    //in that case we don't need to update the reference frame and just use 
    //the old one instead. Worse, normalizing the zero would generate NaN
    //nuking this sample. (There's no branchless version according to the hairy
    //ball theorem)
    new_ref = crosser(dir, new);
    float len = length(new_ref);
    if (len > 1.0e-7) {
        new_ref /= len;
    }
    else {
        //ref and new_ref equal -> do nothing
        new_ref = ref;
        return mat4(1.0);
    }
    float cos_phi = dot(ref, new_ref);
    float sin_phi = dot(crosser(ref, new_ref), dir);

    //create rotation matrix
    float c = 2.0 * cos_phi * cos_phi - 1.0;    //cos(2phi)
    float s = 2.0 * cos_phi * sin_phi;          //sin(2phi)
    //assemble reference frame rotation matrix (column major!)
    return mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0,   c,   s, 0.0,
        0.0,  -s,   c, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

//creates rotation matrix transforming oldRef to newRef, both perpendicular to dir
mat4 matchPolRef(vec3 dir, vec3 oldRef, vec3 newRef) {
    float cos_phi = dot(oldRef, newRef);
    float sin_phi = dot(crosser(oldRef, newRef), dir);

    //create rotation matrix
    float c = 2.0 * cos_phi * cos_phi - 1.0;    //cos(2phi)
    float s = 2.0 * cos_phi * sin_phi;          //sin(2phi)
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

mat4 polarizerMatrix(float p, float s) {
    float att = p*p + s*s;
    float m12 = (p*p - s*s) / att;
    float m33 = (2.0*p*s) / att;
    //assemble matrix (column major!)
    return mat4(
        1.0, m12, 0.0, 0.0,
        m12, 1.0, 0.0, 0.0,
        0.0, 0.0, m33, 0.0,
        0.0, 0.0, 0.0, m33
    );
}

#endif
