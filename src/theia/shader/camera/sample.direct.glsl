layout(local_size_x = 512) in;

#include "result.glsl"
#include "math.glsl"
#include "util/buffers.glsl"
#include "util/sample.glsl"

#include "material.glsl"

#include "rng.glsl"
#include "ray.glsl"
#include "photons.glsl"
#include "camera.glsl"

uniform SamplerParams {
    uvec2 queueAdr;
    uint queueSize;
} samplerParams;

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= samplerParams.queueSize) return;

    //sample light
    float lambda = sampleWavelength(idx, dim);
    vec3 lightDir = sampleUnitSphere(random2D(idx, dim));
    //sample camera
    CameraHit hit;
    CameraSample cam = sampleCamera(lambda, idx, dim);
    BackwardRay ray = createCameraRay(cam, lightDir, lambda, hit);
    
    //save result

    #define _saveFloat(v) floats.values[idx] = (v); idx += samplerParams.queueSize
    #define _saveVec3(v) _saveFloat(v.x); _saveFloat(v.y); _saveFloat(v.z)

    FloatBuffer floats = FloatBuffer(samplerParams.queueAdr);
    _saveVec3(lightDir);
    _saveVec3(cam.position);
    _saveVec3(cam.normal);
    _saveFloat(cam.contrib);

    const uint prependFieldCount = 10;
    const uint offset = 4 * prependFieldCount * samplerParams.queueSize;
    uvec2 queueAdr = shiftAdr(samplerParams.queueAdr, offset);

    saveBackwardRay(
        queueAdr,
        samplerParams.queueSize,
        gl_GlobalInvocationID.x,
        ray,
        hit
    );
}
