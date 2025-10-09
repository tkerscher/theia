#ifndef _LOOKUP_INCLUDE
#define _LOOKUP_INCLUDE

//GLSL has no enums so we have to do it a bit more old fashioned
#define InterpolationMode int
const InterpolationMode INTERPOLATION_LINEAR = 1;
const InterpolationMode INTERPOLATION_CUBIC = 2;
const InterpolationMode INTERPOLATION_STEFFEN = 3;
const InterpolationMode INTERPOLATION_MAKIMA = 4;

//Number of elements padded on both sides per axis
#define LUT_PADDING 1

layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer Table1D {
    InterpolationMode mode;
    int nx; //Number of samples
    float samples[];
};

layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer Table2D {
    InterpolationMode mode;
    int nu, nv; //Number of samples per dim
    float samples[];
    //memory layout is equivalent to float[nu][nv]
    //elements are thus contiguous in the second (v) axis
};

//include interpolation implementation
#include "lookup.linear.glsl"
#include "lookup.cubic.glsl"
#include "lookup.steffen.glsl"

//common code
float lookUp(const Table1D table, float u, float nullValue) {
    if (uint64_t(table) == 0)
        return nullValue;
    
    u = clamp(u, 0.0, 1.0);
    u *= float(table.nx - 1);

    switch(table.mode) {
    case INTERPOLATION_CUBIC:
        return lookUp_cubic(table, u);
    case INTERPOLATION_STEFFEN:
        return lookUp_steffen(table, u);
    default:
        return lookUp_linear(table, u);
    }
}
float lookUp(const Table1D table, float u) {
    return lookUp(table, u, 0.0);
}

float lookUp2D(const Table2D table, float u, float v, float nullValue) {
    if (uint64_t(table) == 0)
        return nullValue;
    
    u = clamp(u, 0.0, 1.0);
    v = clamp(v, 0.0, 1.0);
    u *= float(table.nu - 1);
    v *= float(table.nv - 1);

    switch(table.mode) {
    case INTERPOLATION_CUBIC:
        return lookUp_cubic(table, u, v);
    //Steffen interpolation is only defined for 1D -> default to linear
    // case INTERPOLATION_STEFFEN:
    //     return lookUp_steffen(table, u, v);
    default:
        return lookUp_linear(table, u, v);
    }
}
float lookUp2D(const Table2D table, float u, float v) {
    return lookUp2D(table, u, v, 0.0);
}

#endif
