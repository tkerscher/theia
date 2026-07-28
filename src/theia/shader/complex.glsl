#ifndef _INCLUDE_MATH_COMPLEX
#define _INCLUDE_MATH_COMPLEX

#include "math.glsl"

//implementation of complex numbers and corresponding arithmetic

#define complex vec2
complex cpolar(float r, float phi) { return complex(r * cos(phi), r * sin(phi)); }

#define creal(c) c.x
#define cimag(c) c.y

#define cabs(c) length(c)
float carg(complex c) { return atan(c.y, c.x); }
complex conj(complex c) { return complex(c.x, -c.y); }
float cnorm(complex c) { return dot(c, c); }

// #define cmul(a, b) complex(prodDiff(a.x,b.x,a.y,b.y), a.x*b.y + a*y*b.x)
complex cmul(complex a, complex b) {
    return complex(prodDiff(a.x, b.x, a.y, b.y), dot(a, b.yx));
}
//Based on "A Robust Complex Division in Scilab" by M. Baudin, R.L. Smith (2012)
//arXiv: 1210.4539
precise float _cdiv_internal_compreal(float a, float b, float c, float d, float r, float t) {
    if (r != 0.0) {
        float br = b * r;
        if (br != 0.0)
            return (a + br) * t;
        else
            return a * t + (b * t) * r;
    }
    else {
        return (a + d * (b / c)) * t;
    }
}
complex _cdiv_robust_subinternal(float a, float b, float c, float d) {
    float r = d / c;
    precise float t = 1.0 / (c + d * r);
    return complex(
        _cdiv_internal_compreal(a,b,c,d,r,t),
        _cdiv_internal_compreal(b,-a,c,d,r,t)
    );
}
complex cdiv(complex a, complex b) {
    float f = 1.0;
    if (abs(cimag(b)) > abs(creal(b))) {
        //swap real and imag part for the next step
        a = a.yx;
        b = b.yx;
        f = -1.0;
    }

    complex result = _cdiv_robust_subinternal(creal(a), cimag(a), creal(b), cimag(b));
    result.y *= f;

    return result;
}

complex csin(complex a) { return complex(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y)); }
complex ccos(complex a) { return complex(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y)); }
complex ctan(complex a) { return cdiv(csin(a), ccos(a)); }

complex csqrt(complex a) {
    float r = cabs(a);
    return complex(
        sqrt(0.5 * max(r + creal(a), 0.0)),
        copySignBit(sqrt(0.5 * max(r - creal(a), 0.0)), cimag(a))
        // sqrt(0.5 * max(r - creal(a), 0.0)) * (cimag(a) >= 0.0 ? 1.0 : -1.0)
    );
}
// complex csqrt(complex a) {
//     return cpolar(sqrt(cabs(a)), 0.5 * carg(a));
// }
complex cpow(complex a, float b) {
    float r = pow(cabs(a), b);
    float phi = b * carg(a);
    return complex(r * cos(phi), r * sin(phi));
}
//complex cpow(complex a, complex b)

// sqrt(1 - x^2)
complex cpyth(complex a) {
    return csqrt(cmul(
        complex(1.0 - a.x, -a.y),
        complex(1.0 + a.x, a.y)
    ));
}

complex cexp(complex a) {
    float r = exp(creal(a));
    float phi = cimag(a);
    return complex(r * cos(phi), r * sin(phi));
}
complex clog(complex a) {
    return complex(log(cabs(a)), carg(a));
}

#endif
