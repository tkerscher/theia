#include "result.glsl"
#include "material.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "callback.glsl"
#include "photon.glsl"
#include "source.glsl"
#include "camera.glsl"

//Top level acceleration structure containing the scene
uniform accelerationStructureEXT tlas;

//configure direct sampling
#define DIRECT_SAMPLING_MISS_SHADER_STRIDE 1
#define DIRECT_SAMPLING_MISS_SHADER_OFFSET 0
#define DIRECT_SAMPLING_HIT_SHADER_OFFSET 0
#include "tracer/scene/direct/sample.glsl"

uniform TraceParams {
    uvec2 tlas;
    float maxTime;
    uint batchSize;
} params;

void main() {
    uint dim = 0;
    uint idx = gl_LaunchIDEXT.x;
    #ifndef DISPATCH_INDIRECT
    //we may produce too many rays as we did not dispatch using the exact batch size
    if (idx >= params.batchSize) return;
    #endif

    //sample direct
    sampleDirect(params.tlas, dim);
}
