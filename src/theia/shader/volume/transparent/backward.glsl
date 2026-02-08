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
    //there is no volume interactions. If we did not hit anything,
    //we must have lost the ray
    return hit ? RESULT_CODE_SUCCESS : RESULT_CODE_RAY_LOST;
}

ResultCode applyVolume(
    inout BackwardRay ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
) {
    //same as applyVolumeSampled
    return hit ? RESULT_CODE_SUCCESS : RESULT_CODE_RAY_LOST;   
}

ResultCode sampleVolumeInteraction(
    inout BackwardRay ray,
    uint idx, inout uint dim
) {
    //there is no volume interaction
    return ERROR_CODE_TRACE_ABORT;
}

#endif
