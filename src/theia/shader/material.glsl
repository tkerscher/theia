#ifndef _INCLUDE_MATERIAL
#define _INCLUDE_MATERIAL

#include "lookup.glsl"
//user defined
#include "slots.glsl"

#define SPEED_OF_LIGHT 0.299792458 // m/ns
#define INV_SPEED_OF_LIGHT 3.335640951981521 // ns / m

readonly buffer MediaTable {
    uint stride;    //stride between slots
    //padding to ensure a single uvec2 data entry is not split across two banks
    //might improve performance a bit (maybe)
    uint _padding;

    uvec2 data[];
} mediaTable;

//cant use functions here, unfortunately, so macros it is...
#define loadMediaSlot_Table1D(slot, idx) Table1D(mediaTable.data[mediaTable.stride * MEDIA_SLOT_##slot + idx])
#define loadMediaSlot_Table2D(slot, idx) Table2D(mediaTable.data[mediaTable.stride * MEDIA_SLOT_##slot + idx])
#define loadMediaSlot_vec2(slot, idx) uintBitsToFloat(mediaTable.data[mediaTable.stride * MEDIA_SLOT_##slot + idx])
#define loadMediaSlot_uvec2(slot, idx) mediaTable.data[mediaTable.stride * MEDIA_SLOT_##slot + idx]

//Medium id corresponding to missing (vacuum) medium
const uint VACUUM_MEDIUM_IDX = 0xFFFFFFFFu;
//Checks whether the given idx references vacuum
//We use a special idx to denote this. In order to not copy this magic value
//everywhere, use this function instead.
bool isVacuum(uint mediumIdx) {
    return mediumIdx == VACUUM_MEDIUM_IDX;
}

//util function for mapping wavelength to unit range
float normalize_lambda(uint mediumIdx, float wavelength) {
    vec2 range = loadMediaSlot_vec2(WAVELENGTH_RANGE, mediumIdx);
    return clamp((wavelength - range.x) / (range.y - range.x), 0.0, 1.0);
}

//TODO: Make this user definable
//We'll keep the current constants in memory, so we don't have to constantly
//look up the same values over and over
struct MediumConstants {
    float n;    //refractive index
    float vg;   //group velocity
    float mu_s; //scattering coefficient
    float mu_e; //extinction coefficient
};
MediumConstants lookUpMedium(const uint mediumIdx, float lambda) {
    if (isVacuum(mediumIdx)) {
        return MediumConstants(
            1.0,            //refractive index
            SPEED_OF_LIGHT, //group velocity
            0.0,            //scattering coefficient
            0.0             //extinction coefficient
        );
    }

    //Fetch tables
    Table1D refractive_index = loadMediaSlot_Table1D(REFRACTIVE_INDEX, mediumIdx);
    Table1D group_velocity = loadMediaSlot_Table1D(GROUP_VELOCITY, mediumIdx);
    Table1D absorption = loadMediaSlot_Table1D(ABSORPTION_COEF, mediumIdx);
    Table1D scattering = loadMediaSlot_Table1D(SCATTERING_COEF, mediumIdx);
    //normalize lambda once
    float u = normalize_lambda(mediumIdx, lambda);
    //look coefficients
    float mu_a = lookUp(absorption, u, 0.0);   //absorption
    float mu_s = lookUp(scattering, u, 0.0);   //scattering
    float mu_e = mu_a + mu_s;                   // extinction

    //look up constants in tables; last argument is default value
    return MediumConstants(
        lookUp(refractive_index,  u, 1.0),
        lookUp(group_velocity, u, SPEED_OF_LIGHT),
        mu_s, mu_e
    );
}

readonly buffer MaterialTable {
    uint stride;
    uint _padding;

    uvec2 data[];
} materialTable;

#define loadMaterialSlot_Table1D(slot, idx) Table1D(materialTable.data[materialTable.stride * MATERIAL_SLOT_##slot + idx])
#define loadMaterialSlot_Table2D(slot, idx) Table2D(materialTable.data[materialTable.stride * MATERIAL_SLOT_##slot + idx])
#define loadMaterialSlot_vec2(slot, idx) uintBitsToFloat(materialTable.data[materialTable.stride * MATERIAL_SLOT_##slot + idx])
#define loadMaterialSlot_uvec2(slot, idx) materialTable.data[materialTable.stride * MATERIAL_SLOT_##slot + idx]

//Material flag bits encoding ray intersection behavior
const uint MATERIAL_BLACK_BODY_BIT      = 0x00000001; //Rays gets completely absorbed
const uint MATERIAL_DETECTOR_BIT        = 0x00000002; //Rays reached a target
const uint MATERIAL_LIGHT_SOURCE_BIT    = 0x00000004; //Rays reached a light source
const uint MATERIAL_NO_REFLECT_FWD_BIT  = 0x00000008; //Forward rays never reflect
const uint MATERIAL_NO_REFLECT_BWD_BIT  = 0x00000010; //Backward rays never reflect
const uint MATERIAL_NO_TRANSMIT_FWD_BIT = 0x00000020; //Forward rays never transmit
const uint MATERIAL_NO_TRANSMIT_BWD_BIT = 0x00000040; //Backward rays never transmit
const uint MATERIAL_VOLUME_BORDER_BIT   = 0x00000080; //No geometric effect on Rays

//util function for fetching media and flags
//for material of the given idx, if inwards is true, fetches medium on the inside and
//the corresponding flags for the direction (i.e. inwards), and vice versa.
void queryMaterialSide(uint materialIdx, bool inwards, out uint mediumIdx, out uint flags) {
    uvec2 data;
    if (inwards)
        data = loadMaterialSlot_uvec2(INWARDS, materialIdx);
    else
        data = loadMaterialSlot_uvec2(OUTWARDS, materialIdx);
    mediumIdx = data.x;
    flags = data.y;
}

#endif
