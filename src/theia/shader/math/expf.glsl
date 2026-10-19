/*
  Copyright (c) 2015-2021, Norbert Juffa
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
//https://forums.developer.nvidia.com/t/a-more-accurate-performance-competitive-implementation-of-expf/47528

#ifndef _INCLUDE_MATH_EXPF
#define _INCLUDE_MATH_EXPF

#include "util/float.glsl"

float expf(float a) {
    precise float f, r, j, s, t;
    int i, ia;

    // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
    j = fmaKHR(1.442695, a, 12582912.0) - 12582912.0; // 0x1.715476p0, 0x1.8p23
    f = fmaKHR(j, -6.93145752e-1, a); // -0x1.62e400p-1  // log_2_hi
    f = fmaKHR(j, -1.42860677e-6, f); // -0x1.7f7d1cp-20 // log_2_lo
    i = int(j); // only well-defined for |a| < 104 (see below)
    // approximate r = exp(f) on interval [-log(2)/2, +log(2)/2]
    r =              1.37805939e-3;   // 0x1.694000p-10
    r = fmaKHR(r, f, 8.37312452e-3);  // 0x1.125edcp-7
    r = fmaKHR(r, f, 4.16695364e-2);  // 0x1.555b5ap-5
    r = fmaKHR(r, f, 1.66664720e-1);  // 0x1.555450p-3
    r = fmaKHR(r, f, 4.99999851e-1);  // 0x1.fffff6p-2
    r = fmaKHR(r, f, 1.00000000e+0);  // 0x1.000000p+0
    r = fmaKHR(r, f, 1.00000000e+0);  // 0x1.000000p+0
    // exp(a) = 2**i * r
    ia = (j > 0.0) ? 0 : int(0x83000000u);
    s = intBitsToFloat(0x7f000000 + ia);
    t = intBitsToFloat((i << 23) - ia);
    r = r * s;
    r = r * t;
    // handle special cases: severe overflow / underflow
    if (abs(a) >= 104.0) r = (a > 0.0) ? uintBitsToFloat(0x7F800000u) : 0.0;
    return r;
}

#endif
