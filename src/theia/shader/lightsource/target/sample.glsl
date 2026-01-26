layout(local_size_x = 512) in;

#include "lightsource/target/common.glsl"
#include "util/buffers.glsl"

#include "rng.glsl"
#include "photon.glsl"
#include "target.glsl"

uniform SamplerParams {
    uvec2 queueAdr;
    uint queueSize;
} samplerParams;

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= samplerParams.queueSize) return;

    //sample wavelength
    float wavelength = sampleWavelength(idx, dim);
    //sample target
    LightTargetSample targetSample = sampleLightTarget(wavelength, idx, dim);

    //save result
    uint queueSize = samplerParams.queueSize;
    FloatBuffer floats = FloatBuffer(samplerParams.queueAdr);
    UIntBuffer uints = UIntBuffer(samplerParams.queueAdr);

    #define _saveFloat(v) floats.values[idx] = (v); idx += queueSize
    #define _saveVec3(v) _saveFloat(v.x); _saveFloat(v.y); _saveFloat(v.z)
    #define _saveUInt(v) uints.values[idx] = (v); idx += queueSize

    _saveFloat(wavelength);
    _saveVec3(targetSample.position);
    _saveVec3(targetSample.normal);
    _saveUint(targetSample.mediumIdx);
    _saveFloat(targetSample.contrib);
}
