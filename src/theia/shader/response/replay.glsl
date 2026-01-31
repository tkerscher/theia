layout(local_size_x = 512) in;

//tell ray model we won't be doing any propagation
#define RAY_STATIC

#include "util/buffers.glsl"

#include "rng.glsl"
#include "ray.glsl"
#include "response.glsl"

uniform ReplayParams {
    uvec2 queueAdr;
    uint queueSize;
} replayParams;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint dim = 0;
    //fetch queue count
    Counter counter = Counter(replayParams.queueAdr);
    uvec2 queueAdr = shiftAdr(replayParams.queueAdr, 4);
    //load and process items
    if (idx < counter.count) {
        HitItem item = loadHitItem(queueAdr, replayParams.queueSize, idx);
        response(item, idx, dim);
    }
}
