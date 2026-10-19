layout(local_size_x = 512) in;

#include "target_guide/common.glsl"
#include "util/buffers.glsl"
#include "util/sample.glsl"

#include "rng.glsl"
#include "target_guide.glsl"

uniform SamplerParams {
    uvec2 queueAdr;
    uint queueSize;

    vec3 minObserver;
    vec3 maxObserver;
};

void main() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= queueSize) return;

    //sample observer and direction
    vec3 observer = mix(minObserver, maxObserver, random3D(idx, dim));
    vec3 direction = sampleUnitSphere(random2D(idx, dim));

    //sample target guide
    TargetGuideSample target_sample = sampleTargetGuide(observer, idx, dim);
    TargetGuideSample target_eval = evalTargetGuide(observer, direction);

    //store results in queue
    FloatBuffer floats = FloatBuffer(queueAdr);
    #define _saveFloat(v) floats.values[idx] = (v); idx += queueSize
    #define _saveVec3(v) _saveFloat(v.x); _saveFloat(v.y); _saveFloat(v.z)

    _saveVec3(observer);
    
    _saveVec3(target_sample.dir);
    _saveFloat(target_sample.dist);
    _saveFloat(target_sample.prob);

    _saveVec3(target_eval.dir);
    _saveFloat(target_eval.dist);
    _saveFloat(target_eval.prob);
}
