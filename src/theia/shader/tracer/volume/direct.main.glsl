layout(local_size_x = 512) in;

#include "result.glsl"
#include "material.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "callback.glsl"
#include "photon.glsl"
#include "source.glsl"
#include "camera.glsl"
#include "response.glsl"
#include "volume.glsl"

#include "tracer/volume/direct.glsl"

uniform TraceParams {
    float maxTime;
    uint batchSize;
} params;

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.batchSize) return;

    sampleDirect(params.maxTime, idx, dim);
}
