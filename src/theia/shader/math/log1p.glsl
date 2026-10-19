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

//port from CUDA version:
//https://forums.developer.nvidia.com/t/faster-and-more-accurate-implementation-of-log1pf/40575/14

#ifndef _INCLUDE_MATH_LOGP1
#define _INCLUDE_MATH_LOGP1

#include "util/float.glsl"

//calculates log(1.0+a). Max ulp error 0.87168
float log1p(float a) {
    const float LOG_TWO = 0.693147182;     // 0x1.62e430p-1
    const float TWO_TO_M23 = 1.19209290e-7; // 0x1.0p-23
    const float INF_F = intBitsToFloat(0x7f800000);
    float m, r, s, t, u;
    int e;

    u = a + 1.0;
    e = (floatBitsToInt(u) - floatBitsToInt(0.75)) & int(0xff800000);
    m = intBitsToFloat(floatBitsToInt(a) - e);
    s = intBitsToFloat(floatBitsToInt(4.0) - e); // s' in [2**-126,2**26]
    m = m + fmaKHR(0.25, s, -1.0);
    // approximate log(1+m) on [-0.25, 0.5]
    s = m * m;
    r =              -4.54559326e-2;       // -0x1.746000p-5
    t =               1.05529785e-1;       //  0x1.b04000p-4
    r = fmaKHR(r, s, -1.32279143e-1);      // -0x1.0ee85ep-3
    t = fmaKHR(t, s,  1.44911006e-1);      //  0x1.28c71ap-3
    r = fmaKHR(r, s, -1.66416913e-1);      // -0x1.54d264p-3
    t = fmaKHR(t, s,  1.99886635e-1);      //  0x1.995e2ap-3
    r = fmaKHR(r, s, -2.50001878e-1);      // -0x1.00007ep-2
    r = fmaKHR(t, m, r);
    r = fmaKHR(r, m,  3.33335280e-1);      //  0x1.5555d8p-2
    r = fmaKHR(r, m, -5.00000000e-1);      // -0x1.000000p-1
    r = fmaKHR(r, s, m);
    r = fmaKHR(float(e), LOG_TWO * TWO_TO_M23, r);

    // special cases: a <= -1, +inf, NaN
    bool  ok = (a > -1.0) && (a < INF_F);
    float sp = (a == -1.0) ? intBitsToFloat(0xff800000)         // -inf
             : (a <  -1.0) ? intBitsToFloat(0x7fc00000)         // NaN
             : a;                                               // +inf, NaN
    r = ok ? r : sp;
    r = (abs(a) < 1.49011612e-08) ? a : r;  // |a| < 2^-26: log1p(a) = a (-0, subnormals)
    return r;
}

#endif
