#ifndef _INCLUDE_SURFACE_THIN_FRESNEL_REAL
#define _INCLUDE_SURFACE_THIN_FRESNEL_REAL

/**
 * Calculates the Fresnel power coefficients for a thin layer.
 * Assumes the layer has a real valued refractive index and is thick enough such
 * that no interference effects are observable.
 * Returns the coefficients as vec4: (Rs, Rp, Ts, Tp)
 * where s stands for perpendicular and p for parallel polarization
*/
vec4 fresnel_thinLayer(
    float n0,           //refractive index: incident medium
    float n1,           //refractive index: layer medium
    float n2,           //refractive index: outgoing medium
    float cos0          //cosine incident angle
) {
    //apply Snell's law to all three media
    float sin0 = sqrt(max((1.0 - cos0) * (1.0 + cos0), 0.0));
    float sin1 = sin0 * n0 / n1;
    float sin2 = sin0 * n0 / n2;
    //by clamping to 0.0 we handle total internal reflection
    float cos1 = sqrt(max((1.0 - sin1) * (1.0 + sin1), 0.0));
    float cos2 = sqrt(max((1.0 - sin2) * (1.0 + sin2), 0.0));

    //evaluate Fresnel terms for reflectance
    float rs01 = 2.0 * n0 * cos0 / (n0 * cos0 + n1 * cos1) - 1.0;
    float rp01 = 2.0 * n1 * cos0 / (n0 * cos1 + n1 * cos0) - 1.0;
    float rs12 = 2.0 * n1 * cos1 / (n1 * cos1 + n2 * cos2) - 1.0;
    float rp12 = 2.0 * n2 * cos1 / (n1 * cos2 + n2 * cos1) - 1.0;
    //convert amplitude to power by squaring the coefficients
    rs01 *= rs01;
    rp01 *= rp01;
    rs12 *= rs12;
    rp12 *= rp12;

    //calculate total reflectance
    //
    //            T01 * T10 * R12
    // R = R01 + -----------------
    //             1 - R10 * R12
    //
    float Ts01 = 1.0 - rs01;
    float Tp01 = 1.0 - rp01;
    float denom_s = max(1.0 - rs01 * rs12, 0.0);
    float Rs = rs01;
    if (denom_s > 0.0) Rs += Ts01 * Ts01 * rs12 / denom_s; //avoid 0/0
    float denom_p = max(1.0 - rp01 * rp12, 0.0);
    float Rp = rp01;
    if (denom_p > 0.0) Rp += Tp01 * Tp01 * rp12 / denom_p; //avoid 0/0

    //return result
    return vec4(Rs, Rp, 1.0 - Rs, 1.0 - Rp);
}

#endif
