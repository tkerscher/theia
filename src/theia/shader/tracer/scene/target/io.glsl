#ifndef _INCLUDE_TRACER_SCENE_FORWARD_IO
#define _INCLUDE_TRACER_SCENE_FORWARD_IO

uniform TraceParams {
    uvec2 tlas;
    PropagationParams propagation;
    int targetId;
    uint batchSize;
} params;

struct TraceData{
    TRACE_RAY ray;
    uint dim;
    ResultCode result;
};

struct NeeData {
    TRACE_RAY ray;
    float weight;
    uint dim;
};

#endif
