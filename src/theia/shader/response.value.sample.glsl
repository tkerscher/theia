#ifndef _INCLUDE_RESPONSE_SAMPLE_VALUE
#define _INCLUDE_RESPONSE_SAMPLE_VALUE

layout(scalar) writeonly buffer ValueQueueOut { float value[]; } valueQueueOut;

void initResponse() {}

void response(HitItem item) {
    uint idx = gl_GlobalInvocationID.x;
    valueQueueOut.value[idx] = responseValue(item);
}

void finalizeResponse() {}

#endif
