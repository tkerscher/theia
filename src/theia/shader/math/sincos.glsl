/*
  Copyright (c) 2016-2019 Norbert Juffa
  All rights reserved.

  Redistribution and use in source and binary forms, with or without 
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright 
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//port from CUDA version:
//https://forums.developer.nvidia.com/t/a-faster-and-more-accurate-implementation-of-sincosf/44620

#ifndef _INCLUDE_MATH_SINCOS
#define _INCLUDE_MATH_SINCOS

#include "util/float.glsl"

//191 bits of 2/PI for Payne-Hanek style argument reduction
const uint _SINCOS_TWO_OVER_PI[6] = uint[](
    0x517CC1B7u,
    0x27220A94u,
    0xFE13ABE8u,
    0xFA9A6EE0u,
    0x6DB14ACCu,
    0x9E21C820u
);

float _sincos_red_slowpath(float a, out int quadrant) {
    uint bits = floatBitsToUint(a);
    uint ia = ((bits & 0x007FFFFFu) | 0x00800000u) << 8;
    int e = int((bits >> 23) & 0xFFu) - 127;

    //compute the product x * 2/pi in 2.62 fixed-point format
    int i = min(e >> 5, 3); //protect against GLSL looser handling of inf/NaN
    e = e & 31;

    uint hi  = i > 0 ? _SINCOS_TWO_OVER_PI[i-1] : 0u;
    uint mid = _SINCOS_TWO_OVER_PI[i+0];
    uint lo  = _SINCOS_TWO_OVER_PI[i+1];
    uint tmp = _SINCOS_TWO_OVER_PI[i+2];

    if (e != 0) {
        hi  = (hi  << e) | (mid >> (32 - e));
        mid = (mid << e) | (lo  >> (32 - e));
        lo  = (lo  << e) | (tmp >> (32 - e));
    }

    // p = (unsigned long long int)ia * lo;
    // p = (unsigned long long int)ia * mid + (p >> 32);
    // p = ((unsigned long long int)(ia * hi) << 32) + p;
    uint phi, plo, carry, lo_hi, lo_lo;
    umulExtended(ia, lo, lo_hi, lo_lo);
    umulExtended(ia, mid, phi, plo);
    plo = uaddCarry(plo, lo_hi, carry);
    phi = phi + carry + ia * hi;

    int q = int(phi >> 30);             //integral portion = quadrant<1:0>
    phi = phi & 0x3FFFFFFFu;            //fraction
    if ((phi & 0x20000000u) != 0u) {    //fraction >= 0.5
        phi = phi - 0x40000000u;        //fraction - 1.0
        q = q + 1;
    }

    //compute remainder of x / (pi/2)
    // d = (double)(long long int)p;
    // d = d * 3.4061215800865545e-19; // 0x1.921fb54442d18p-62 // pi/2 * 0x1.p-62
    // r = (float)d;
    //take |p| (two's complement) ...
    bool neg = (phi & 0x80000000u) != 0u;
    if (neg) {
        plo = ~plo + 1u;
        phi = ~phi + (plo == 0u ? 1u : 0u);
    }
    //... normalize it to |p| ~ m * 2^sh with the leading one in bit 31 of m
    //(truncation error < 2^-31 relative), |p| <= 2^61 -> msb(phi) <= 29 ...
    uint m;
    int sh;
    if (phi != 0u) {
        int msb = findMSB(phi);
        m = (phi << (31 - msb)) | (plo >> (msb + 1));
        sh = msb + 1;
    }
    else {
        int msb = findMSB(plo); //-1 for plo == 0 -> m = 0
        m = msb >= 0 ? plo << (31 - msb) : 0u;
        sh = msb - 31;
    }
    //... and multiply with pi/2 in double-float arithmetic
    float mh = float(m & 0xFFFFFF00u); //exact: 24 significant bits
    float ml = float(m & 0x000000FFu); //exact
    precise float ph = mh * 1.57079637;          // 0x1.921fb6p+0  // pio2_hi
    precise float pl = fmaKHR(mh, 1.57079637, -ph); //exact error of ph
    pl = fmaKHR(mh, -4.37113883e-8, pl);            // -0x1.777a5cp-25 // pio2_lo
    pl = fmaKHR(ml, 1.57079637, pl);
    precise float r = ldexp(ph + pl, sh - 62);
    if (neg) r = -r;

    if (a < 0.0) {
        r = -r;
        q = -q;
    }

    quadrant = q;
    return r;
}

//Argument reduction for trigonometric functions that reduces the argument
//to the interval [-PI/4, +PI/4] and also returns the quadrant. It returns
//-0.0 for an input of -0.0
float _sincos_red(float a, float switch_over, out int q) {
    //FMA-enhanced Cody-Waite style reduction. W. J. Cody and W. Waite,
    //"Software Manual for the Elementary Functions", Prentice-Hall 1980
    precise float j = fmaKHR(a, 0.636619747, 12582912.0); // 2/pi, 0x1.8p+23
    q = floatBitsToInt(j); //only the two LSBs are needed
    j = j - 12582912.0; // 0x1.8p+23
    precise float r;
    r = fmaKHR(j, -1.57079601e+00, a); // -0x1.921fb0p+00 // pio2_high
    r = fmaKHR(j, -3.13916473e-07, r); // -0x1.5110b4p-22 // pio2_mid
    r = fmaKHR(j, -5.39030253e-15, r); // -0x1.846988p-48 // pio2_low
    if (abs(a) > switch_over) {
        r = _sincos_red_slowpath(a, q);
    }
    return r;
}

//Approximate sine on [-PI/4,+PI/4]. Maximum ulp error = 0.64196
//Returns -0.0 for an argument of -0.0
//Polynomial approximation based on T. Myklebust, "Computing accurate
//Horner form approximations to special functions in finite precision
//arithmetic", http://arxiv.org/abs/1508.03211, retrieved on 8/29/2016
float _sincos_sin_poly(float a, float s) {
    precise float r, t;
    r =               2.86567956e-6;   //  0x1.80a000p-19
    r = fmaKHR(r, s, -1.98559923e-4);  // -0x1.a0690cp-13
    r = fmaKHR(r, s,  8.33338592e-3);  //  0x1.111182p-07
    r = fmaKHR(r, s, -1.66666672e-1);  // -0x1.555556p-03
    t = fmaKHR(a, s, 0.0); // ensure -0 is passed trough
    r = fmaKHR(r, t, a);
    return r;
}

//Approximate cosine on [-PI/4,+PI/4]. Maximum ulp error = 0.87444
float _sincos_cos_poly(float s) {
    precise float r;
    r =               2.44677067e-5;   //  0x1.9a8000p-16
    r = fmaKHR(r, s, -1.38877297e-3);  // -0x1.6c0efap-10
    r = fmaKHR(r, s,  4.16666567e-2);  //  0x1.555550p-05
    r = fmaKHR(r, s, -5.00000000e-1);  // -0x1.000000p-01
    r = fmaKHR(r, s,  1.00000000e+0);  //  0x1.000000p+00
    return r;
}

//Computes sine and cosine of a simultaneously.
//Maximum ulp error sin = 1.49241, maximum ulp error cos = 1.49510
vec2 sincos(float a) {
    precise float x = a * 0.0 + a; // inf -> NaN
    int i;
    float r = _sincos_red(x, 71476.0625, i);

    //compute sine and cosine simultaneously, based on quadrant
    precise float s = r * r;
    precise float c = _sincos_cos_poly(s);
    s = _sincos_sin_poly(r, s);
    if ((i & 2) != 0) {
        s = 0.0 - s; // don't change "sign" of NaNs or create negative zeros
        c = 0.0 - c; // don't change "sign" of NaNs or create negative zeros
    }
    if ((i & 1) != 0) {
        precise float t = 0.0 - s; // don't change "sign" of NaNs or create negative zeros
        s = c;
        c = t;
    }
    //|a| < 2^-26: sin(a) = a. Passing a through also keeps -0 on devices that
    //do not preserve signed zeros in arithmetic (allowed by Vulkan without the
    //SignedZeroInfNanPreserve execution mode), as done in expm1() and log1p()
    return vec2((abs(a) < 1.49011612e-08) ? a : s, c);
}

#endif
