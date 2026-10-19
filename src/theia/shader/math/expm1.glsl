/*
  Copyright (c) 2015-2023, Norbert Juffa
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

//port from CUDA versions:
//https://forums.developer.nvidia.com/t/a-faster-and-more-accurate-implementation-of-expm1f/48085/2

#ifndef _INCLUDE_MATH_EXPM1
#define _INCLUDE_MATH_EXPM1

#include "util/float.glsl"

//Computes exp(a)-1.0. Maximum ulp error = 0.997458
float expm1(float a) {
    float f, j, r, s, t, u, v;
    precise float x, y;  // keep the compiler from folding (t - y) - s to 0
    int i;

    // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
    j = fmaKHR(1.442695, a, 12582912.0);   // 0x1.715476p0, 0x1.8p23
    i = floatBitsToInt(j);              // trailing bits contain integer
    j = j - 12582912.0;                 // 0x1.8p23
    f = fmaKHR(j, -6.93145752e-1, a);      // -0x1.62e400p-1  // log_2_hi
    f = fmaKHR(j, -1.42860677e-6, f);      // -0x1.7f7d1cp-20 // log_2_lo

    // approximate r = exp(f)-1 on interval [-log(2)/2, +log(2)/2]
    s = f * f;
    if (a == 0.0) s = a;                // ensure -0 is passed through
    r =              1.97350979e-4;        // 0x1.9de000p-13
    r = fmaKHR(r, f, 1.39309070e-3);       // 0x1.6d30bcp-10
    r = fmaKHR(r, f, 8.33343994e-3);       // 0x1.1111f6p-7
    r = fmaKHR(r, f, 4.16668020e-2);       // 0x1.55559ep-5
    r = fmaKHR(r, f, 1.66666716e-1);       // 0x1.55555cp-3
    r = fmaKHR(r, f, 4.99999970e-1);       // 0x1.fffffep-2
    // if i == 0, expm1(a) = z
    // if i == 1, expm1(a) = 2*(r*(f*f)+f+0.5)
    // if (i < 0) || (i > 1) expm1(a) = 2*((r*(f*f)+f)*t-0.5+t)
    u = (j == 1.0) ? (f + 0.5) : f;
    v = fmaKHR(r, s, u);
    s = 0.5;
    t = intBitsToFloat((i << 23) + floatBitsToInt(s));
    y = t - s;
    x = (t - y) - s;                    // double-float canonicalization of difference
    r = fmaKHR(v, t, x) + y;
    r = r + r;
    if (j == 0.0) r = v;
    if (j == 1.0) r = v + v;

    // handle severe overflow and underflow
    if (abs(a - 1.0) > 88.0) {
        r = exp2(a);
        r = fmaKHR(r, r, -1.0);
    }
    r = (abs(a) < 1.49011612e-08) ? a : r;  // |a| < 2^-26: expm1(a) = a (-0, subnormals)
    return r;
}

#endif
