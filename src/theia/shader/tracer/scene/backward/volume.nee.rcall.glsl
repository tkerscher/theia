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

#include "tracer/propagate/common.glsl"

uniform TraceParams {
    uvec2 tlas;
    PropagationParams propagation;
    uint batchSize;
} params;


struct VolumeNeeData {
    BackwardRay ray;
    float tMax;
    uint dim;
};
layout(location = 3) callableDataInEXT VolumeNeeData volumeNeeData;

//we need to have a callable for each volume shader, including the non-scattering
//one -> turn callable in a no-op in that case

#ifndef VOLUME_MODEL_NO_SCATTERING

#include "tracer/scene/backward/volume.nee.glsl"

void main() {    
    //resample camera hit (so we do not need to cache it in memory)
    uint reDim = CAMERA_SAMPLE_RNG_DIM;
    float lambdaContrib;
    float lambda = sampleWavelength(lambdaContrib, gl_LaunchIDEXT.x, reDim);
    CameraHit camHit;
    sampleCameraRay(lambda, camHit, gl_LaunchIDEXT.x, reDim);

    estimateInScattering(volumeNeeData.ray, camHit, volumeNeeData.tMax, volumeNeeData.dim);
}

#else

//no scattering supported

void main() {}

#endif
