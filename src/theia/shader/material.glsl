#ifndef _INCLUDE_MATERIAL
#define _INCLUDE_MATERIAL

#include "lookup.glsl"
//user defined
#include "slots.glsl"

#define SPEED_OF_LIGHT 0.299792458 // m/ns
#define INV_SPEED_OF_LIGHT 3.335640951981521 // ns / m

//Medium id corresponding to missing (vacuum) medium
const uint VACUUM_MEDIUM_IDX = 0xFFFFFFFFu;
//Checks whether the given idx references vacuum
//We use a special idx to denote this. In order to not copy this magic value
//everywhere, use this function instead.
bool isVacuum(uint mediumIdx) {
    return mediumIdx == VACUUM_MEDIUM_IDX;
}

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

float lookUpMediaTable1D_imp(uint slotIdx, uint mediumIdx, float u, float def) {
    if (isVacuum(mediumIdx)) return def;

    Table1D table = Table1D(mediaTable.data[mediaTable.stride * slotIdx + mediumIdx]);
    return lookUp(table, u, def);
}
#define lookUpMediaTable1D(slot, idx, u, def) lookUpMediaTable1D_imp(MEDIA_SLOT_##slot, idx, u, def)

//util function for mapping wavelength to unit range
// float normalize_lambda(uint mediumIdx, float wavelength) {
//     if (isVacuum(mediumIdx)) return 0.0;

//     vec2 range = loadMediaSlot_vec2(WAVELENGTH_RANGE, mediumIdx);
//     return clamp((wavelength - range.x) / (range.y - range.x), 0.0, 1.0);
// }

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
const uint MATERIAL_BLACK_BODY_BIT          = 0x00000001; //Rays gets completely absorbed
const uint MATERIAL_DETECTOR_BIT            = 0x00000002; //Rays reached a target
const uint MATERIAL_LIGHT_SOURCE_BIT        = 0x00000004; //Rays reached a light source
const uint MATERIAL_NO_REFLECT_FWD_BIT      = 0x00000008; //Forward rays never reflect
const uint MATERIAL_NO_REFLECT_BWD_BIT      = 0x00000010; //Backward rays never reflect
const uint MATERIAL_NO_TRANSMIT_FWD_BIT     = 0x00000020; //Forward rays never transmit
const uint MATERIAL_NO_TRANSMIT_BWD_BIT     = 0x00000040; //Backward rays never transmit
// const uint MATERIAL_VOLUME_BORDER_BIT       = 0x00000080; //No geometric effect on Rays (deprecated)
const uint MATERIAL_SKIP_MISMATCH_TEST_BIT  = 0x00000100; //Skip media mismatch test
const uint MATERIAL_TRANSMIT_HIT_BIT        = 0x00000200; //Transmit hits before detection

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
