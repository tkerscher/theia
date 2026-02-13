#include "result.glsl"
#include "material.glsl"
//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "callback.glsl"
#include "camera.glsl"
#include "source.glsl"
#include "response.glsl"
#include "volume.glsl"

#include "tracer/propagate/backward.glsl"
#include "tracer/propagate/combine.glsl"

uniform TraceParams {
    uvec2 tlas;
    PropagationParams propagation;
} params;

struct NeeData {
    BackwardRay ray;
    uint dim;
};
layout(location = 1) rayPayloadInEXT NeeData neeData;

void main() {
    //since the RNG is deterministic, instead of putting the sampled light source
    //into the payload, we simply "resample" the light source with the exact same
    //RNG state as earlier to produce the exact same sample
    ForwardRay source = sampleLight(
        neeData.ray.position,
        vec3(0.0),
        neeData.ray.wavelength,
        neeData.ray.mediumIdx,
        gl_LaunchIDEXT.x,
        neeData.dim
    );
    //the same goes for the camera ray
    uint dim = CAMERA_SAMPLE_RNG_DIM;
    CameraHit camHit;
    sampleCameraRay(
        neeData.ray.wavelength,
        camHit,
        gl_LaunchIDEXT.x,
        dim
    );
    //TODO: we actually do not the whole camera ray but are only interested in
    //      the camera hit. Should we introduce a new function to the camera API
    //      to recreate only the camera hit?

    //create hit by combining light and camera ray
    HitItem hit;
    ResultCode result = combineRays(
        neeData.ray,
        source,
        camHit,
        params.propagation,
        hit,
        gl_LaunchIDEXT.x, neeData.dim
    );
    if (result >= 0) {
        response(hit, gl_LaunchIDEXT.x, neeData.dim);
    }
}
