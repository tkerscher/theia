//Check expected macros
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef PATH_LENGTH
#error "PATH_LENGTH not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "math.glsl"
#include "ray.propagate.glsl"
#include "tracer.photon.queue.glsl"

layout(scalar) uniform DispatchParams {
    uint batchSize;
};

#include "tracer.volume.photon.common.glsl"

#include "wavelengthsource.common.glsl"
#include "lightsource.common.glsl"
//user provided code
#include "source.glsl"
#include "photon.glsl"

ForwardRay sampleRay(uint idx, inout uint dim) {
    WavelengthSample photon = sampleWavelength(idx, dim);
    Medium medium = Medium(params.medium);
    MediumConstants constants = lookUpMedium(medium, photon.wavelength);
    SourceRay lightRay = sampleLight(photon.wavelength, constants, idx, dim);
    ForwardRay ray = createRay(lightRay, medium, constants, photon);

    //in photon tracing, we want contrib to be the survival chance (1.0 - absorption)
    //thus we have to set the initial contrib to 1.0
    //mathematically, we multiply with the normalization constant here
    ray.state.lin_contrib = 1.0;
    ray.state.log_contrib = 0.0;

    return ray;
}

void traceMain() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batchSize)
        return;
    
    //sample ray
    ForwardRay ray = sampleRay(idx, dim);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);
    //discard ray if inside target
    if (isOccludedByTarget(ray.state.position)) {
        onEvent(ray, ERROR_CODE_TRACE_ABORT, idx, 0);
        return;
    }

    //trace loop
    traceLoop(ray, idx, dim, 0, true);
}

void main() {
    initResponse();
    traceMain();
    finalizeResponse();
}
