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
    //there is no volume interaction, if we did not hit anything,
    //the ray must have been lost
    return hit ? RESULT_CODE_SUCCESS : RESULT_CODE_RAY_LOST;
}

ResultCode applyVolume(
    inout ForwardRay ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
) {
    //same as applyVolumeSampled
    return hit ? RESULT_CODE_SUCCESS : RESULT_CODE_RAY_LOST;
}

ResultCode sampleVolumeInteraction(
    inout ForwardRay,
    uint idx, inout uint dim
) {
    //there is no volume interaction
    return ERROR_CODE_TRACE_ABORT;
}

#endif
