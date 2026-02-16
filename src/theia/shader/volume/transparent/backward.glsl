#ifndef _INCLUDE_VOLUME_MODEL_TRANSPARENT_BACKWARD
#define _INCLUDE_VOLUME_MODEL_TRANSPARENT_BACKWARD

float sampleInteractionLength(
    const BackwardRay ray,
    uint idx, inout uint dim
) {
    return 1.0 / 0.0; //+inf
}

ResultCode applyVolumeSampled(
    inout BackwardRay ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
) {
    //there is no volume interaction
    return RESULT_CODE_SUCCESS;
}

ResultCode applyVolume(
    inout BackwardRay ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
) {
    //same as applyVolumeSampled
    return RESULT_CODE_SUCCESS;
}

ResultCode sampleVolumeInteraction(
    inout BackwardRay ray,
    uint idx, inout uint dim
) {
    //there is no volume interaction
    return ERROR_CODE_TRACE_ABORT;
}

//There is no scattering in transparent media
#define VOLUME_MODEL_NO_SCATTERING

#endif
