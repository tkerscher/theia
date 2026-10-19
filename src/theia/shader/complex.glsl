#ifndef _INCLUDE_MATH_COMPLEX
#define _INCLUDE_MATH_COMPLEX

#include "math.glsl"
#include "math/expf.glsl"
#include "math/expm1.glsl"
#include "math/hyperbolic.glsl"
#include "math/log1p.glsl"
#include "math/sincos.glsl"

/******************************* internal helper ******************************/

//error-free transformation: a + b = s + e exactly (Knuth's TwoSum)
vec2 _ctwoSum(float a, float b) {
    precise float s = a + b;
    precise float t = s - a;
    precise float e = (a - (s - t)) + (b - t);
    return vec2(s, e);
}

//x^2 + sign * y^2 - 1 calculated in double-float precision
//to avoid catastrophic cancelation
float _cSqSumMinusOne(float x, float y, float s) {
    precise float p = x * x;
    precise float pe = fmaKHR(x, x, -p);
    precise float q = s * y * y;
    precise float qe = fmaKHR(s * y, y, -q);
    vec2 t = _ctwoSum(p, q);
    vec2 u = _ctwoSum(t.x, -1.0);
    precise float result = u.x + (((t.y + u.y) + pe) + qe);
    return result;
}

/******************************* complex numbers ******************************/

#define complex vec2

#define creal(c) (c).x
#define cimag(c) (c).y

complex cpolar(float r, float phi) {
    return r * sincos(phi).yx;
}

float cabs(complex c) {
    //use a numerical more stable calculation than the built-in length()
    vec2 a = abs(c);
    float m = max(a.x, a.y);
    if (m == 0.0 || isinf(m)) return m;
    int e;
    frexp(m, e);
    a = ldexp(a, ivec2(-e)); //max in [0.5,1)
    return ldexp(sqrt(dot(a, a)), e);
}

float carg(complex c) {
    if (cimag(c) == 0.0) {
        return copySignBit(signBit(creal(c)) > 0.0 ? 0.0 : PI, cimag(c));
    }
    return atan(cimag(c), creal(c));
}

complex conj(complex c) { return complex(c.x, -c.y); }
float cnorm(complex c) { return dot(c, c); }

complex cmul(complex a, complex b) {
    return complex(
        prodDiff(a.x, b.x, a.y, b.y),   //a.x*b.x - a.y*b.y
        prodDiff(a.x, b.y, -a.y, b.x)   //a.x*b.y + a.y*b.x
    );
}

complex cdiv(complex a, complex b) {
    //a / b = a * conj(b) / |b|^2
    int ea, eb;
    frexp(max(abs(creal(a)), abs(cimag(a))), ea);
    frexp(max(abs(creal(b)), abs(cimag(b))), eb);
    a = ldexp(a, ivec2(-ea));
    b = ldexp(b, ivec2(-eb));
    return ldexp(cmul(a, conj(b)) / cnorm(b), ivec2(ea - eb));
}

complex csqrt(complex a) {
    float x = creal(a), y = cimag(a);
    //scale a exactly by 2^-2k to max(|x|,|y|) in [0.5, 2), so that neither
    //overflow nor (flushed) subnormals occur; the result is scaled by 2^k
    int e;
    frexp(max(abs(x), abs(y)), e);
    int k = e >> 1; //floor(e / 2)
    float rho = sqrt(0.5 * (abs(ldexp(x, -2 * k)) + cabs(ldexp(a, ivec2(-2 * k)))));
    if (rho == 0.0) return complex(0.0, y); //a = 0
    //|y| / (2 * rho * 2^k); y scaled separately, as it may be tiny compared to x
    float other = ldexp(ldexp(abs(y), -k) / rho, -1);
    rho = ldexp(rho, k);
    return x >= 0.0
        ? complex(rho, copySignBit(other, y))
        : complex(other, copySignBit(rho, y));
}

//sqrt(1 - z^2) with 1 - z^2 = 1 - x^2 + y^2 - 2ixy
complex cpyth(complex z) {
    float x = creal(z), y = cimag(z);
    return csqrt(complex(-_cSqSumMinusOne(x, y, -1.0), -2.0 * x * y));
}

complex cexp(complex a) {
    return cpolar(expf(creal(a)), cimag(a));
}

complex clog(complex a) {
    vec2 b = abs(a);
    float m = max(b.x, b.y);
    float n = min(b.x, b.y);
    float re;
    if (m >= 0.5 && m <= 2.0) {
        //log|a| = 0.5 * log(1 + (|a|^2 - 1)) avoids the loss of precision
        //of log() for |a| ~ 1
        re = 0.5 * log1p(_cSqSumMinusOne(m, n, 1.0));
    }
    else {
        re = m > 1e38 ? log(cabs(0.25 * a)) + log(4.0) : log(cabs(a));
    }
    return complex(re, carg(a));
}

//0^b: 0 for Re(b) > 0, 1 for b = 0, otherwise a pole (returns inf)
complex _cpow_zero(complex b) {
    if (creal(b) > 0.0) return complex(0.0, 0.0);
    if (b == complex(0.0, 0.0)) return complex(1.0, 0.0);
    return complex(uintBitsToFloat(0x7F800000u), 0.0);
}
//principal value of a^b
complex cpow(complex a, float b) {
    if (a == complex(0.0, 0.0)) return _cpow_zero(complex(b, 0.0));
    return cexp(b * clog(a));
}
complex cpow(complex a, complex b) {
    if (a == complex(0.0, 0.0)) return _cpow_zero(b);
    return cexp(cmul(b, clog(a)));
}

complex csin(complex a) {
    return sincos(creal(a)) * sinhcoshf(cimag(a)).yx;
}
complex ccos(complex a) {
    vec2 sc = sincos(creal(a));
    vec2 sch = sinhcoshf(cimag(a));
    return complex(sc.y * sch.y, -sc.x * sch.x);
}
complex ctan(complex a) {
    //tan(x + iy) = (sin x cos x + i sinh y cosh y) / (cos^2 x + sinh^2 y)
    //equals (sin 2x + i sinh 2y) / (cos 2x + cosh 2y), but without its cancellation
    //in the denominator, nor overflow of csin(a) / ccos(a) for large |y|
    vec2 sc = sincos(creal(a));
    float y = cimag(a);
    if (abs(y) > 10.0) {
        //sinh(y)^2 ~ cosh(y)^2 ~ exp(2|y|) / 4 -> imag part is +/-1 within float precision
        return complex(4.0 * sc.x * sc.y * expf(-2.0 * abs(y)), copySignBit(1.0, y));
    }
    vec2 sch = sinhcoshf(y);
    return complex(sc.x * sc.y, sch.x * sch.y) / fmaKHR(sc.y, sc.y, sch.x * sch.x);
}

#endif
