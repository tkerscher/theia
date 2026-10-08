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

    vec3 origin;
    vec3 direction;
    float maxStep;
} samplerParams;

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= samplerParams.count) return;

    float wavelength = sampleWavelength(idx, dim);
    //sample ray origin if not specified
    vec3 origin = samplerParams.origin;
    if (any(isnan(origin))) {
        origin = vec3(
            mix(-DIM, DIM, random(idx, dim)),
            mix(-DIM, DIM, random(idx, dim)),
            mix(-DIM, DIM, random(idx, dim))
        );
    }
    //sample ray direction if not specified
    vec3 direction = samplerParams.direction;
    if (any(isnan(direction))) {
        direction = sampleHemisphere(random2D(idx, dim));
    }

    //sample light
    #ifdef LightNeeContext
    LightNeeContext context;
    #endif
    float tSampled = sampleLightNeeStepSize(
        origin, direction,
        samplerParams.maxStep,
        wavelength, samplerParams.mediumIdx,
        #ifdef LightNeeContext
        context,
        #endif
        idx, dim
    );
    //eval
    float tStep = random(idx, dim) * samplerParams.maxStep;
    float residualProb = 0.5 + random(idx, dim);
    ForwardRay ray = sampleLightMIS(
        origin + tStep * direction,
        direction,
        tStep,
        wavelength,
        samplerParams.mediumIdx,
        residualProb,
        #ifdef LightNeeContext
        context,
        #endif
        idx, dim
    );

    //save results

    #define _saveFloat(v) floats.values[idx] = (v); idx += samplerParams.count
    #define _saveVec3(v) _saveFloat(v.x); _saveFloat(v.y); _saveFloat(v.z)

    FloatBuffer floats = FloatBuffer(samplerParams.queueAdr);
    _saveVec3(origin);
    _saveVec3(direction);
    _saveFloat(wavelength);
    _saveFloat(tSampled);
    _saveFloat(tStep);
    _saveFloat(residualProb);

    #undef _saveFloat
    #undef _saveVec3

    const uint prependFieldCount = 10;
    const uint offset = 4 * prependFieldCount * samplerParams.count;
    uvec2 queueAdr = shiftAdr(samplerParams.queueAdr, offset);
    saveForwardRay(queueAdr, samplerParams.count, gl_GlobalInvocationID.x, ray);
}
