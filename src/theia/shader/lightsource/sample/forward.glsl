layout(local_size_x = 512) in;

#include "result.glsl"
#include "material.glsl"
#include "util/buffers.glsl"

//user provided code
#include "rng.glsl"
#include "ray.glsl"
#include "photon.glsl"
#include "light.glsl"

uniform SampleParams {
    uint count;
    uint baseCount;

    uvec2 queueAdr;
    uint queueSize;
} sampleParams;

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= sampleParams.count)
        return;

    //sample light
    ForwardRay ray = sampleLight(idx + sampleParams.baseCount, dim);
    //save light in queue
    saveForwardRay(sampleParams.queueAdr, sampleParams.queueSize, idx, ray);
}
