//Check expected macros
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "ray.glsl"
#include "ray.propagate.glsl"

layout(scalar) uniform DispatchParams {
    uint batchSize;
};

layout(scalar) uniform TraceParams {
    uvec2 medium;

    PropagateParams propagation;
} params;

#ifdef USE_SCENE

//Top level acceleration structure containing the scene
uniform accelerationStructureEXT tlas;

#endif

//define global medium
#define USE_GLOBAL_MEDIUM
Medium getMedium() {
    return Medium(params.medium);
}
#include "ray.medium.glsl"

#include "ray.combine.glsl"
#include "ray.response.glsl"
#include "result.glsl"

#include "lightsource.common.glsl"
#include "camera.common.glsl"
#include "response.common.glsl"
#include "wavelengthsource.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "camera.glsl"
#include "source.glsl"
#include "response.glsl"
#include "photon.glsl"

#include "callback.util.glsl"

#include "tracer.direct.common.glsl"

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    
    initResponse();
    if (idx < batchSize) {
        sampleDirect(idx, dim, Medium(params.medium), params.propagation);
    }
    finalizeResponse();
}
