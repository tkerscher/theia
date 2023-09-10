#ifndef _MATERIAL_INCLUDE
#define _MATERIAL_INCLUDE

#include "lookup.glsl"

#define SPEED_OF_LIGHT 299792458.0

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
    Table1D phase;          //scattering phase function
    Table1D phase_sampling; //used for sampling
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
    float mu_a; //absorption coefficient
    float mu_s; //scattering coefficient
};

MediumConstants lookUpMedium(const Medium medium, float lambda) {
    //null pointer means vacuum
    if (uint64_t(medium) == 0) {
        return MediumConstants(
            1.0,            //refractive index
            SPEED_OF_LIGHT, //group velocity
            0.0,            //absorption coefficient
            0.0             //scattering coefficient
        );
    }

    //normalize lambda once
    float u = normalize_lambda(medium, lambda);

    //look up constants in tables; last argument is default value
    return MediumConstants(
        lookUp(medium.n,    u, 1.0),            //refractive index
        lookUp(medium.vg,   u, SPEED_OF_LIGHT), //group velocity
        lookUp(medium.mu_a, u, 0.0),            //absorption coefficient
        lookUp(medium.mu_s, u, 0.0)             //scattering coefficient
    );
}

/////////////////////////////////// MATERIAL ///////////////////////////////////

const uint32_t MATERIAL_ABSORBER_BIT        = 0x00000001; //Rays gets absorped
const uint32_t MATERIAL_TARGET_BIT          = 0x00000002; //Rays reached a target
const uint32_t MATERIAL_SOURCE_BIT          = 0x00000004; //Rays reached a light source
const uint32_t MATERIAL_METALLIC_BIT        = 0x00000008; //Consider reflections only

//Materials are assigned to geometries, which with their polygon faces define
//an inside and an outside using their winding order
layout(buffer_reference, scalar, buffer_reference_align=8) buffer Material {
    Medium inside;
    Medium outside;

    // uint32_t flags;
    // uint32_t padding; //to fit the 8 byte alignment
};

#endif
