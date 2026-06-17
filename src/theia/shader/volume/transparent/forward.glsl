#ifndef _INCLUDE_VOLUME_MODEL_TRANSPARENT_FORWARD
#define _INCLUDE_VOLUME_MODEL_TRANSPARENT_FORWARD

float sampleInteractionLength(
    const ForwardRay ray,
    uint idx, inout uint dim    
) {
    return 1.0 / 0.0; //+inf
}

ResultCode applyVolumeSampled(
    inout ForwardRay ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
) {
    //there is no volume interaction
    return RESULT_CODE_SUCCESS;
}

ResultCode applyVolume(
    inout ForwardRay ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
) {
    //same as applyVolumeSampled
    return RESULT_CODE_SUCCESS;
}

ResultCode sampleVolumeInteraction(
    inout ForwardRay,
    uint idx, inout uint dim
) {
    //there is no volume interaction -> keep on straight
    return RESULT_CODE_SUCCESS;
}

//There is no scattering in transparent media
#define VOLUME_MODEL_NO_SCATTERING

#endif
