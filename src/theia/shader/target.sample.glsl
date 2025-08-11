layout(local_size_x = 32) in;

#include "target.common.glsl"
#include "util.sample.glsl"

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
writeonly buffer ResultBuffer{ Result r[]; };

layout(scalar, push_constant) uniform Push {
    vec3 minObserver;
    vec3 maxObserver;
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= BATCH_SIZE) return;
    uint dim = 0;

    //sample observer and direction
    vec3 observer = vec3(
        mix(minObserver.x, maxObserver.x, random(i, dim)),
        mix(minObserver.y, maxObserver.y, random(i, dim)),
        mix(minObserver.z, maxObserver.z, random(i, dim))
    );
    vec3 direction = sampleUnitSphere(random2D(i, dim));

    //sample target
    r[i] = Result(
        observer,
        direction,
        checkSample(observer, sampleTarget(observer, i, dim)),
        checkIntersect(observer, direction, intersectTarget(observer, direction)),
        isOccludedByTarget(observer)
    );
}
