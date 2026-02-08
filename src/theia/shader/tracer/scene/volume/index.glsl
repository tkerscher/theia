#ifndef _INCLUDE_TRACER_VOLUME_MAP
#define _INCLUDE_TRACER_VOLUME_MAP

//Mapping from medium index to volume model index
readonly buffer MediumMap { uint mediumMap[]; };

//we will always map transparent to #0
const uint TRANSPARENT_IDX = 0u;

uint getVolumeIdx(uint mediumIdx) {
    if (mediumIdx == VACUUM_MEDIUM_IDX)
        return TRANSPARENT_IDX;
    else
        return mediumMap[mediumIdx];
}

#endif
