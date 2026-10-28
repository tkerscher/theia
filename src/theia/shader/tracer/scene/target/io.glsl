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
    //Id of the photon/path. This is NOT gl_LaunchIDEXT.x: a tracer with wavefront
    //compaction relaunches surviving paths at a different launch index, so the id
    //has to travel with the ray.
    uint idx;
    uint dim;
    ResultCode result;
};

struct NeeData {
    TRACE_RAY ray;
    float weight;
    //Id of the photon/path
    uint idx;
    uint dim;
};

#endif
