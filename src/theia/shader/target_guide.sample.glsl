layout(local_size_x = 32) in;

#include "target_guide.common.glsl"
#include "util.sample.glsl"

#include "rng.glsl"
#include "target_guide.glsl"

struct Result {
    vec3 observer;
    TargetGuideSample targ;
    TargetGuideSample eval;
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

    //sample target guide
    r[i] = Result(
        observer,
        sampleTargetGuide(observer, i, dim),
        evalTargetGuide(observer, direction)
    );
}
