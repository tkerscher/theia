//check expected macros
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef PATH_LENGTH
#error "PATH_LENGTH not defined"
#endif

//configure tracing
#define SCENE_TRAVERSE_DISABLE_MIS 1
#define SCENE_TRAVERSE_FORWARD 1

layout(local_size_x = BLOCK_SIZE) in;

#include "ray.propagate.glsl"
#include "scene.intersect.glsl"
#include "scene.traverse.glsl"
#include "tracer.photon.queue.glsl"

#include "wavelengthsource.common.glsl"
#include "lightsource.common.glsl"
#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "source.glsl"
#include "photon.glsl"
#include "response.glsl"

#include "callback.util.glsl"

uniform DispatchParams {
    uint batchSize;
};

uniform TraceParams {
    int targetId;
    uvec2 sceneMedium;

    PropagateParams propagation;
} params;

ForwardRay sampleRay(uint idx, inout uint dim) {
    WavelengthSample photon = sampleWavelength(idx, dim);
    Medium medium = Medium(params.sceneMedium);
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

    //trace loop
    for (uint i = 1; i <= PATH_LENGTH; ++i) {
        //trace ray
        SurfaceHit hit;
        ResultCode result = trace(
            ray, hit,
            params.targetId,
            idx, dim,
            params.propagation,
            true //allow response
        );
        if (result >= 0) {
            result = processInteraction(
                ray, hit, params.targetId,
                idx, dim,
                params.propagation,
                true, //allow response
                false //last
            );
        }

        //sample absorption throwing away photons early to increase performance
        if (result >= 0) {
            //for correct absorption sampling, we need a normalized stokes vector
            #ifdef POLARIZATION
            ray.state.lin_contrib *= ray.stokes.x;
            ray.stokes /= ray.stokes.x;
            #endif
            //sample absorption
            if (getContrib(ray) <= random(idx, dim)) {
                result = RESULT_CODE_RAY_ABSORBED;
            }
            else {
                //reset contrib
                ray.state.lin_contrib = 1.0;
                ray.state.log_contrib = 0.0;
            }
        }

        onEvent(ray, result, idx, i);
        //stop codes are negative
        if (result < 0)
            return;
    }

    //save rays and load them in the next iteration to get fully converged work groups again  
    saveRay(ray, idx);
}

void main() {
    initResponse();
    traceMain();
    finalizeResponse();
}
