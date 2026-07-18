#ifndef _INCLUDE_SURFACE_THIN_FRESNEL_COMPLEX
#define _INCLUDE_SURFACE_THIN_FRESNEL_COMPLEX

#include "complex.glsl"

/**
 * Calculates the Fresnel power coefficients for a thin layer of given
 * thickness with complex refractive index.
 * Returns the coefficients as vec4: (Rs, Rp, Ts, Tp)
 * where s stands for perpendicular and p for parallel polarization
*/
vec4 fresnel_thinLayer(
    float n0, /* k0=0 */    //refractive index: incident medium
    float n1, float k1,     //refractive index: layer medium
    float n2, /* k2=0 */    //refractive index: outgoing medium
    float d,                //layer thickness [nm]
    float lambda,           //wavelength [nm]
    float cos0              //cosine incident angle
) {
    //assemble complex refractive indices
    complex N0 = complex(n0, 0.0); //incident
    complex N1 = complex(n1, k1);  //layer
    complex N2 = complex(n2, 0.0); //outgoing

    //apply Snell's law to all three media (note: complex sines and cosines)
    float sin0 = sqrt(max((1.0 - cos0) * (1.0 + cos0), 0.0));
    complex cos1 = cpyth(cdiv(N0 * sin0, N1));
    complex cos2 = cpyth(cdiv(N0 * sin0, N2));
    //There's a mathematical branch cut along the negative axis.
    //Floats can distinguish the branches via signed zeros (+/- 0.0).
    //Due to finite precision we might end up at either branch near zero.
    //Unfortunately, this sign can propagate causing evanescent waves to explode
    //in magnitude instead of decaying.
    //-> ensure the imaginary parts are always non-negative
    cimag(cos1) = abs(cimag(cos1));
    cimag(cos2) = abs(cimag(cos2));

    //calculate fresnel coefficients
    //incident -> layer (0 -> 1)
    complex ts01 = cdiv(2.0 * N0 * cos0, N0 * cos0 + cmul(N1, cos1));
    complex rs01 = ts01 - complex(1.0, 0.0);
    complex tp01 = cdiv(2.0 * N0 * cos0, N1 * cos0 + n0 * cos1);
    complex rp01 = cdiv(2.0 * N1 * cos0, N1 * cos0 + n0 * cos1) - complex(1.0, 0.0);
    //layer -> outgoing (1 -> 2)
    complex ts12 = cdiv(2.0 * cmul(N1, cos1), cmul(N1, cos1) + n2 * cos2);
    complex rs12 = ts12 - complex(1.0, 0.0);
    complex tp12 = cdiv(2.0 * cmul(N1, cos1), n2 * cos1 + cmul(N1, cos2));
    complex rp12 = cdiv(2.0 * n2 * cos1, n2 * cos1 + cmul(N1, cos2)) - complex(1.0, 0.0);

    //combined reflectance and transmittance coefficients
    //
    //            t_01 * t_12 * exp(i*b)
    // t_012 = -----------------------------
    //          1 + r_01 * r_12 * exp(2i*b)
    //
    //            r_01 + r_12 * exp(2i*b)
    // r_012 = -----------------------------
    //          1 + r_01 * r_12 * exp(2i*b)
    //
    // b = 2pi * d / lambda * N1 * cos_1
    //
    complex beta = TWO_PI * d / lambda * cmul(N1, cos1);
    complex exp_beta = cexp(complex(-beta.y, beta.x));          //exp(i*beta)
    complex exp_beta2 = cexp(2.0 * complex(-beta.y, beta.x));   //exp(2i*beta)
    complex denom_s = complex(1.0, 0.0) + cmul(cmul(rs01, rs12), exp_beta2);
    complex denom_p = complex(1.0, 0.0) + cmul(cmul(rp01, rp12), exp_beta2);
    complex ts012 = cdiv(cmul(cmul(ts01, ts12), exp_beta), denom_s);
    complex tp012 = cdiv(cmul(cmul(tp01, tp12), exp_beta), denom_p);
    complex rs012 = cdiv(rs01 + cmul(rs12, exp_beta2), denom_s);
    complex rp012 = cdiv(rp01 + cmul(rp12, exp_beta2), denom_p);

    //finally, we can calculate reflectance and transmissivity
    //absorptivity is then the missing energy: A = 1 - R - T
    float Rs = cnorm(rs012);
    float Rp = cnorm(rp012);
    float t_scale = (n2 * creal(cos2)) / (n0 * cos0); //use real part of cos2 to handle total internal reflection
    float Ts = t_scale * cnorm(ts012);
    float Tp = t_scale * cnorm(tp012);

    //return result
    return vec4(Rs, Rp, Ts, Tp);
}

#endif
