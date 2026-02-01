layout(local_size_x = 512) in;

#include "target/common.glsl"
#include "util/buffers.glsl"
#include "util/sample.glsl"

#include "rng.glsl"
#include "target.glsl"

struct CompressedTargetSample {
    vec3 position;
    vec3 normal;
    float prob;
    bool valid;
    int error;
};

CompressedTargetSample checkSample(vec3 observer, TargetSample ts) {
    //short hand
    mat3 m = ts.worldToObj;
    vec3 o = ts.offset;
    //check for correct transformation
    float expD = distance(ts.position, observer);
    vec3 expPos = m * ts.position + o;
    vec3 expNrm = normalize(m * ts.normal);
    int error = 0;
    error += int(abs(expD - ts.dist) > 1e-5);
    error += int(length(expPos - ts.objPosition) > 1e-5) << 1;
    error += int(length(expNrm - ts.objNormal) > 1e-5) << 2;
    
    return CompressedTargetSample(
        ts.position,
        ts.normal,
        ts.prob,
        ts.valid,
        error
    );
}

CompressedTargetSample checkIntersect(vec3 observer, vec3 direction, TargetSample ts) {
    //common check
    CompressedTargetSample c = checkSample(observer, ts);

    //additionally check direction
    vec3 sampleDir = normalize(ts.position - observer);
    c.error += (int(length(sampleDir - direction) > 1e-5) << 3);

    return c;
}

struct Result {
    vec3 observer;
    vec3 direction;
    CompressedTargetSample target; //sample
    CompressedTargetSample hit;
    bool occluded;
};

uniform SamplerParams {
    uvec2 queueAdr;
    uint queueSize;

    vec3 minObserver;
    vec3 maxObserver;
} samplerParams;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= samplerParams.queueSize) return;
    uint dim = 0;

    //sample observer and direction
    vec3 observer = vec3(
        mix(samplerParams.minObserver.x, samplerParams.maxObserver.x, random(i, dim)),
        mix(samplerParams.minObserver.y, samplerParams.maxObserver.y, random(i, dim)),
        mix(samplerParams.minObserver.z, samplerParams.maxObserver.z, random(i, dim))
    );
    vec3 direction = sampleUnitSphere(random2D(i, dim));
    //sample target
    CompressedTargetSample sampl = checkSample(observer, sampleTarget(observer, i, dim));
    CompressedTargetSample hit = checkIntersect(observer, direction, intersectTarget(observer, direction));

    //save results
    uint idx = gl_GlobalInvocationID.x;
    uint queueSize = samplerParams.queueSize;
    FloatBuffer floats = FloatBuffer(samplerParams.queueAdr);
    UIntBuffer uints = UIntBuffer(samplerParams.queueAdr);
    IntBuffer ints = IntBuffer(samplerParams.queueAdr);

    #define _saveInt(v) ints.values[idx] = (v); idx += queueSize
    #define _saveUInt(v) uints.values[idx] = (v); idx += queueSize
    #define _saveFloat(v) floats.values[idx] = (v); idx += queueSize
    #define _saveVec3(v) _saveFloat(v.x); _saveFloat(v.y); _saveFloat(v.z)

    _saveVec3(observer);
    _saveVec3(direction);

    _saveVec3(sampl.position);
    _saveVec3(sampl.normal);
    _saveFloat(sampl.prob);
    _saveUInt(uint(sampl.valid));
    _saveInt(sampl.error);

    _saveVec3(hit.position);
    _saveVec3(hit.normal);
    _saveFloat(hit.prob);
    _saveUInt(uint(hit.valid));
    _saveInt(hit.error);
    
    _saveUInt(uint(isOccludedByTarget(observer)));
}
