#ifndef _INCLUDE_SURFACE_FRESNEL
#define _INCLUDE_SURFACE_FRESNEL

//Evaluates the Fresnel reflectance for unpolarized light hitting the interface
//between two dielectric media..
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

#endif
