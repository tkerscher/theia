layout(local_size_x = 32) in;

#include "random.glsl"
#include "random.gamma.glsl"

layout(scalar) writeonly buffer Result{ float x[]; };

layout(push_constant) uniform PushConstant {
    float alpha;
    float lambda;
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint dim = 0;
    x[i] = sampleGamma(alpha, lambda, i, dim);
}
