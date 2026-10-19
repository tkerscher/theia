#ifndef _INCLUDE_MATH_HYPERBOLIC
#define _INCLUDE_MATH_HYPERBOLIC

#include "math.glsl"
#include "math/expf.glsl"
#include "math/expm1.glsl"

#define _LOG_FLT_MAX uintBitsToFloat(0x42B17217u)

//sign * exp(x) / 2
float _expo2f(float x, float s) {
    const float kln2 = 162.88958740234375;     // 0x1.45c778p+7
    float scale = uintBitsToFloat(0x7A000000u); // 2^117 = 2^(k/2)
    precise float r = expf(x - kln2) * (s * scale) * scale; //precise: order matters
    return r;
}

//sinh(a) for 0 <= a <= 1 using minimax polynom; max error 0.730 ulp
float _sinh_poly(float ax) {
    precise float s = ax * ax;
    precise float p = 2.816949973e-6;   // 0x1.7a15aap-19
    p = fmaKHR(p, s,  1.983619004e-4);  // 0x1.9ffebcp-13
    p = fmaKHR(p, s,  8.333341219e-3);  // 0x1.111122p-7
    p = fmaKHR(p, s,  1.666666716e-1);  // 0x1.555556p-3
    p = fmaKHR(ax * s, p, ax);
    return p;
}

//cosh(a) for 0 <= a <= 1 using minimax polynom; max error 0.885 ulp
float _cosh_poly(float ax) {
    precise float s = ax * ax;
    precise float q = 2.546724318e-5;   // 0x1.ab44fap-16
    q = fmaKHR(q, s,  1.388343167e-3);  // 0x1.6bf222p-10
    q = fmaKHR(q, s,  4.166685417e-2);  // 0x1.5555bap-5
    q = fmaKHR(q, s,  4.999999702e-1);  // 0x1.fffffep-2
    q = fmaKHR(s, q,  1.0);
    return q;
}

float sinhf(float x) {
    float ax = abs(x);
    float r;
    if (ax < 1.0) {
        r = _sinh_poly(ax);
    }
    else if (ax < _LOG_FLT_MAX) {
        //sinh(x) = (e - 1/e) / 2 = (t + t / (t + 1)) / 2 with t = e - 1
        float t = expm1(ax);
        r = 0.5 * (t + t / (t + 1.0));
    }
    else {
        //|x| >= logf(FLT_MAX) or NaN
        r = _expo2f(ax, 1.0);
    }
    //bit operation: keeps -0 even on devices not preserving signed zeros
    return copySignBit(r, x);
}

float coshf(float x) {
    float ax = abs(x);
    if (ax < 1.0)
        return _cosh_poly(ax);
    if (ax < _LOG_FLT_MAX) {
        float e = expf(ax);
        return 0.5 * (e + 1.0 / e);
    }
    //|x| >= logf(FLT_MAX) or NaN
    return _expo2f(ax, 1.0);
}

//computes sinh and cosh in one go; max error 1.458 ulp
vec2 sinhcoshf(float x) {
    float ax = abs(x);
    float r, ch, sh;
    if (ax < 1.0) {
        r = _sinh_poly(ax);
        ch = _cosh_poly(ax);
    }
    else if (ax < _LOG_FLT_MAX) {
        //e = t + 1 = exp(|x|): sinh = (t + t/e) / 2, cosh = (e + 1/e) / 2
        float t = expm1(ax);
        float e = t + 1.0;
        r = 0.5 * (t + t / e);
        ch = 0.5 * (e + 1.0 / e);
    }
    else {
        //|x| >= logf(FLT_MAX) or NaN
        r = _expo2f(ax, 1.0);
        ch = r;
    }
    sh = copySignBit(r, x);

    return vec2(sh, ch);
}

#undef _LOG_FLT_MAX

#endif
