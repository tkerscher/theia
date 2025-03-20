#include "rng.glsl"
#include "scatter.volume.glsl"

layout(local_size_x = 32) in;

struct Query {
    vec3 dir;
    uvec2 medium;
};
layout(scalar) readonly buffer Input{
    Query queries[];
};

struct Result{
    vec3 dir;
    float prob;
};
layout(scalar) writeonly buffer Output{
    Result results[];
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    Query q = queries[i];
    
    float prob;
    vec3 dir = scatter(Medium(q.medium), q.dir, random2D_s(i, 0), prob);

    results[i] = Result(dir, prob);
}
