#include "scatter.volume.glsl"

layout(local_size_x = 32) in;

struct Query{
    vec3 inDir;
    vec3 scatterDir;
    uvec2 medium;
};
readonly buffer Input{
    Query queries[];
};

writeonly buffer Output{
    float p[];
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    Query q = queries[i];
    p[i] = scatterProb(Medium(q.medium), q.inDir, q.scatterDir);
}
