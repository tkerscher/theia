layout(local_size_x = 512) in;

#include "result.glsl"
#include "material.glsl"
#include "util/buffers.glsl"
#include "util/sample.glsl"

//user provided code
#include "rng.glsl"
#include "ray.glsl"
#include "photon.glsl"
#include "source.glsl"

uniform SamplerParams {
    uvec2 queueAdr;
    uint count;

    uint mediumIdx;

    vec3 observer;
} samplerParams;

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= samplerParams.count) return;

    float wavelength = sampleWavelength(idx, dim);
    //sample observer position if not specified
    vec3 observer = samplerParams.observer;
    if (any(isnan(observer))) {
        observer = vec3(
            mix(-DIM, DIM, random(idx, dim)),
            mix(-DIM, DIM, random(idx, dim)),
            mix(-DIM, DIM, random(idx, dim))
        );
    }
    //sample observer normal
    vec3 normal = sampleHemisphere(random2D(idx, dim));

    //sample light
    ForwardRay ray = sampleLight(
        observer,
        normal,
        wavelength,
        samplerParams.mediumIdx,
        idx, dim
    );

    //save result

    #define _saveFloat(v) floats.values[idx] = (v); idx += samplerParams.count
    #define _saveVec3(v) _saveFloat(v.x); _saveFloat(v.y); _saveFloat(v.z)

    FloatBuffer floats = FloatBuffer(samplerParams.queueAdr);
    _saveVec3(observer);
    _saveVec3(normal);
    _saveFloat(wavelength);

    #undef _saveFloat
    #undef _saveVec3

    const uint prependFieldCount = 7;
    const uint offset = 4 * prependFieldCount * samplerParams.count;
    uvec2 queueAdr = shiftAdr(samplerParams.queueAdr, offset);
    saveForwardRay(queueAdr, samplerParams.count, gl_GlobalInvocationID.x, ray);
}
