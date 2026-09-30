#ifndef _INCLUDE_SURFACE_FRESNEL
#define _INCLUDE_SURFACE_FRESNEL

/**
 * Evaluates the Fresnel reflectance for unpolarized light hitting the interface
 * between two dielectric media under the incidence angle described by `cos_i`
 * (cosine of the angle between the incident ray and the (micro-facet) normal).
 *
 * By clamping `cos_o` to zero, total internal reflection is handled accurately
 * (the reflectance evaluates to 1.0).
 */
float fresnelReflectance(float cos_i, float n_i, float n_o) {
    //calculate outgoing angle (Snell's law)
    float sin_i = sqrt(max(1.0 - cos_i*cos_i, 0.0));
    float sin_o = sin_i * n_i / n_o;
    //by clamping cos_o to 0.0 we accurately handle total internal reflection
    float cos_o = sqrt(max(1.0 - sin_o*sin_o, 0.0));

    //evaluate Fresnel terms for reflectance
    float r_s = (n_i * cos_i - n_o * cos_o) / (n_i * cos_i + n_o * cos_o);
    float r_p = (n_o * cos_i - n_i * cos_o) / (n_o * cos_i + n_i * cos_o);
    return 0.5 * (r_s*r_s + r_p*r_p);
}

/**
 * Evaluates the Fresnel reflectance for unpolarized light going from a
 * dielectric medium with real refractive index `n_i` into a conductor with
 * complex refractive index `n + i*k`, under the incidence angle described by
 * `cos_i`. Uses the exact (real-arithmetic) form of the conductor Fresnel
 * equations.
 */
float fresnelConductor(float cos_i, float n_i, float n, float k) {
    cos_i = clamp(cos_i, 0.0, 1.0);
    float eta = n / n_i;
    float etak = k / n_i;

    float cos2 = cos_i * cos_i;
    float sin2 = 1.0 - cos2;
    float eta2 = eta * eta;
    float etak2 = etak * etak;

    float t0 = eta2 - etak2 - sin2;
    float a2plusb2 = sqrt(max(t0*t0 + 4.0*eta2*etak2, 0.0));
    float t1 = a2plusb2 + cos2;
    float a = sqrt(max(0.5*(a2plusb2 + t0), 0.0));
    float t2 = 2.0 * a * cos_i;
    float Rs = (t1 - t2) / (t1 + t2);

    float t3 = cos2 * a2plusb2 + sin2 * sin2;
    float t4 = t2 * sin2;
    float Rp = Rs * (t3 - t4) / (t3 + t4);

    return 0.5 * (Rs + Rp);
}

#endif
