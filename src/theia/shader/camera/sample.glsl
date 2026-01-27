layout(local_size_x = 512) in;

#include "result.glsl"
#include "material.glsl"

//user provided code
#include "rng.glsl"
#include "ray.glsl"
#include "photon.glsl"
#include "camera.glsl"

uniform SampleParams {
    uvec2 queueAdr;
    uint queueSize;

    uint count;
    uint baseCount;
} sampleParams;

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= sampleParams.count)
        return;
    idx += sampleParams.baseCount;
    
    //sample wavelength
    float wavelength = sampleWavelength(idx, dim);
    //sample camera
    CameraHit hit;
    BackwardRay ray = sampleCameraRay(wavelength, hit, idx, dim);
    //save sample
    saveBackwardRay(
        sampleParams.queueAdr, sampleParams.queueSize,
        gl_GlobalInvocationID.x,
        ray, hit
    );
}
