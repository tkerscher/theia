#ifndef _MATERIAL_INCLUDE
#define _MATERIAL_INCLUDE

#include "lookup.glsl"

#define SPEED_OF_LIGHT 0.299792458 // m/ns
#define INV_SPEED_OF_LIGHT 3.335640951981521 // ns / m

//////////////////////////////////// MEDIUM ////////////////////////////////////

layout(buffer_reference, scalar, buffer_reference_align=8) buffer Medium {
    //valid range shared by all tables
    //  -> normalize only once
    float lambda_min;
    float lambda_max;

    //Tables are defined with respect to vacuum wavelength normalized to
    //the above specified range

    //wave properties
    Table1D n;              //refractive index
    Table1D vg;             //group velocity

    //material constants
    Table1D mu_a;           //absorption coefficient
    Table1D mu_s;           //scattering coefficient
    Table1D log_phase;      //scattering phase function
    Table1D phase_sampling; //used for sampling
    //phase/mueller matrix
    Table1D phase_m12;
    Table1D phase_m22;
    Table1D phase_m33;
    Table1D phase_m34;
};

//util function to make things more readable
float normalize_lambda(const Medium medium, float lambda) {
    return clamp((lambda - medium.lambda_min) / (medium.lambda_max - medium.lambda_min), 0.0, 1.0);
}

//We'll keep the current constants in memory, so we don't have to constantly
//look up the same values over and over
struct MediumConstants {
    float n;    //refractive index
    float vg;   //group velocity
    float mu_s; //scattering coefficient
    float mu_e; //extinction coefficient
};

MediumConstants lookUpMedium(const Medium medium, float lambda) {
    //null pointer means vacuum
    if (uint64_t(medium) == 0) {
        return MediumConstants(
            1.0,            //refractive index
            SPEED_OF_LIGHT, //group velocity
            0.0,            //scattering coefficient
            0.0             //extinction coefficient
        );
    }

    //normalize lambda once
    float u = normalize_lambda(medium, lambda);
    //look coefficients
    float mu_a = lookUp(medium.mu_a, u, 0.0);   //absorption
    float mu_s = lookUp(medium.mu_s, u, 0.0);   //scattering
    float mu_e = mu_a + mu_s;                   // extinction

    //look up constants in tables; last argument is default value
    return MediumConstants(
        lookUp(medium.n,  u, 1.0),              //refractive index
        lookUp(medium.vg, u, SPEED_OF_LIGHT),   //group velocity
        mu_s, mu_e
    );
}

/////////////////////////////////// MATERIAL ///////////////////////////////////

//Material flag bits encoding ray intersection behavior
const uint MATERIAL_BLACK_BODY_BIT      = 0x00000001; //Rays gets completely absorbed
const uint MATERIAL_DETECTOR_BIT        = 0x00000002; //Rays reached a target
const uint MATERIAL_LIGHT_SOURCE_BIT    = 0x00000004; //Rays reached a light source
const uint MATERIAL_NO_REFLECT_FWD_BIT  = 0x00000008; //Forward rays never reflect
const uint MATERIAL_NO_REFLECT_BWD_BIT  = 0x00000010; //Backward rays never reflect
const uint MATERIAL_NO_TRANSMIT_FWD_BIT = 0x00000020; //Forward rays never transmit
const uint MATERIAL_NO_TRANSMIT_BWD_BIT = 0x00000040; //Backward rays never transmit
const uint MATERIAL_VOLUME_BORDER_BIT   = 0x00000080; //No geometric effect on Rays

//Materials are assigned to geometries, which with their normals pointing
//outwards define an inside and an outside. Both directions can be assigned
//flags to define the behavior of rays intersecting the geometry.
layout(buffer_reference, scalar, buffer_reference_align=8) buffer Material {
    Medium inside;
    Medium outside;

    uint flagsInwards;
    uint flagsOutwards;
};

#endif
